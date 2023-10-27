from pyexpat import model
import torch
import os
from tqdm import tqdm 

from dataset import MedicalDatasetRaw
from model import Model

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from train import Config,accuracy

if __name__ == "__main__":
    
    opt = Config("medical-test")
    opt.bert_path = "result/models/medical-01/9-0.25-99.pt"

    ## data
    e_d = MedicalDatasetRaw(opt, 'project_data/test', False)
    eval_dataset = DataLoader(e_d, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
    opt.class_num = e_d.get_class_num()

    ## result
    log_path = 'result/logs/%s' % opt.model_name
    os.makedirs(log_path, exist_ok=True)
    logger = SummaryWriter(log_dir=log_path)

    ## model
    # model = Model(opt)
    model = torch.load(opt.bert_path)
    model.eval().to(opt.device)
    loss_f = torch.nn.CrossEntropyLoss()

    ### begin to test ###
    print('begin to test...')
    bar = enumerate(eval_dataset)
    length = len(eval_dataset)
    bar = tqdm(bar, total=length)
    total_acc = 0
    total_loss = 0
    total_batch = 0
    with torch.no_grad():
        for i, batch in bar:
            total_batch += 1
            inputs = batch[0]
            labels = batch[1][:,0].long()
            outputs = model(inputs)
            # loss = F.cross_entropy(outputs, labels.t()) # not one hot
            loss = loss_f(outputs, labels) # not one hot
            acc = accuracy(outputs, labels, topk = [3])[0]
            total_acc = total_acc + acc.to('cpu').detach().numpy()[0]
            total_loss = total_loss + loss.item()

            logger.add_scalar('loss', loss.item(), total_batch)
            logger.add_scalar('acc', acc.to('cpu').detach().numpy()[0], total_batch)
            
        total_loss = total_loss / total_batch
        total_acc = total_acc / total_batch
        print(f'Train total Loss: {total_loss},  Train total Acc {total_acc}')