import torch
import os
import numpy as np
from tqdm import tqdm 
from torch.autograd import Variable

from dataset import MedicalDatasetRaw
from model import Model
from transformers import BertTokenizer
from optimization import BertAdam

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

def accuracy(output, target, topk=(1, )):       
    # output.shape (batch_size, num_classes), target.shape (bs, )

    """Computes the accuracy over the k top predictions for the specified values of k"""

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def fgsm_attack(inputs, epsilon, data_grad):
    # Get the sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create perturbed image by adjusting each pixel of the input image
    perturbed_input = inputs + epsilon * sign_data_grad
    # Return perturbed image
    return perturbed_input

def eval():
    pass

class Config(object):
    
    def __init__(self, name):

        self.model_name = name                                                        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                      
        self.num_epochs = 5                                           
        self.batch_size = 8
        self.n_cpu = 0                                          
        self.pad_size = 64  # process to fix length
        # self.pad_size = 0
        self.learning_rate = 1e-5                                 
        self.bert_path = '/data/wuchengyan/chip/macbert'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.class_num = 6

if __name__ == "__main__":
    opt = Config("medical-03")

    ## data
    #t_d = MedicalDatasetRaw(opt, 'project_data/train', False)
    t_d = MedicalDatasetRaw(opt, '/data/wuchengyan/Chinese-Text-Classification-main/project_data/train.csv')
    e_d = MedicalDatasetRaw(opt, '/data/wuchengyan/Chinese-Text-Classification-main/project_data/test.csv')
    train_dataset = DataLoader(t_d, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
    opt.class_num = t_d.get_class_num()

    ## result
    log_path = 'result/logs/%s' % opt.model_name
    model_path = 'result/models/%s' % opt.model_name

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    logger = SummaryWriter(log_dir=log_path)

    ## model
    model = Model(opt)
    model.train().to(opt.device)
    loss_f = torch.nn.CrossEntropyLoss()

    ## init rand
    np.random.seed(666)
    torch.manual_seed(666)
    torch.cuda.manual_seed_all(666)

    ## optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(parameters, lr=opt.learning_rate, warmup=0.05, t_total=len(train_dataset) * opt.num_epochs)

    #### start to train ####
    print('Start to train ...')

    total_step = 0
    eval_step = 0

    for epoch in range(opt.num_epochs):
        print('epoch :', epoch)
        bar = enumerate(train_dataset)
        length = len(train_dataset)
        bar = tqdm(bar, total=length)
        total_acc = 0
        total_loss = 0
        total_batch = 0
        for i, batch in bar:
            total_step += 1
            total_batch += 1
            inputs = batch[0]
            labels = batch[1][:, 0].long()
            inputs, labels = inputs.to(opt.device).long(), labels.to(opt.device)  # Change to long()
            inputs.requires_grad = True  # Set requires_grad directly on the tensor

            # Forward pass
            outputs = model(inputs)
            loss = loss_f(outputs, labels)

            # Zero gradients, backward pass, and optimize
            model.zero_grad()
            loss.backward()

            # FGSM attack
            data_grad = inputs.grad.data
            epsilon = 0.02
            perturbed_inputs = fgsm_attack(inputs, epsilon, data_grad)

            # Re-forward pass with perturbed inputs
            outputs = model(perturbed_inputs)
            loss = loss_f(outputs, labels)

            acc = accuracy(outputs, labels, topk=[3])[0]
            total_acc = total_acc + acc.to('cpu').detach().numpy()[0]
            total_loss = total_loss + loss.item()
            optimizer.step()

            logger.add_scalar('loss', loss.item(), total_step)
            logger.add_scalar('acc', acc.to('cpu').detach().numpy(), total_step)

        total_loss = total_loss / total_batch
        total_acc = total_acc / total_batch
        print(f'Train total Loss: {total_loss},  Train total Acc {total_acc}')
        torch.save(model, model_path + f'/{epoch}-{total_loss}-{total_acc}.pt')  # save the whole model
