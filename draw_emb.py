from pyexpat import model
import torch
import os
from tqdm import tqdm 
import torch.nn.functional as F
from dataset import MedicalDatasetRaw
from model import Model
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from train import Config,accuracy

if __name__ == "__main__":
    
    opt = Config("medical-test")
    opt.bert_path = "/data/wuchengyan/Chinese-Text-Classification-main/result/models/medical-03/3-0.14390793319946776-95.7.pt"

    ## data
    #e_d = MedicalDatasetRaw(opt, 'project_data/test', False)
    e_d = MedicalDatasetRaw(opt, '/data/wuchengyan/Chinese-Text-Classification-main/csv/pred.csv')
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
    # Modify the code after the for loop for embedding visualization
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for i, batch in bar:
            total_batch += 1
            inputs = batch[0]
            labels = batch[1][:,0].long()
            outputs = model(inputs)
            all_embeddings.append(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # loss = F.cross_entropy(outputs, labels.t()) # not one hot
            loss = loss_f(outputs, labels) # not one hot
            acc = accuracy(outputs, labels, topk = [3])[0]
            total_acc = total_acc + acc.to('cpu').detach().numpy()[0]
            total_loss = total_loss + loss.item()

            logger.add_scalar('loss', loss.item(), total_batch)
            logger.add_scalar('acc', acc.to('cpu').detach().numpy()[0], total_batch)
            # 打印输出每个样本的预测结果
            probabilities = F.softmax(outputs, dim=1)
            for j in range(len(labels)):
                print(f"Sample {total_batch * opt.batch_size + j + 1}: Predicted Probabilities: {probabilities[j]}, Actual: {labels[j]}")
            
        total_loss = total_loss / total_batch
        total_acc = total_acc / total_batch
        print(f'Train total Loss: {total_loss},  Train total Acc {total_acc}')
    # Convert the list of embeddings to a single numpy array
    all_embeddings = np.concatenate(all_embeddings, axis=0)

    # Apply t-SNE for dimensionality reduction
    tsne_embeddings = TSNE(n_components=2).fit_transform(all_embeddings)

    # Plot the t-SNE embeddings
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_embeddings[:, 0], y=tsne_embeddings[:, 1], hue=all_labels, palette="viridis")
    plt.title('t-SNE Visualization of Sentence Embeddings')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(title='Labels', loc='upper right')
    plt.savefig('sentence_embeddings_visualization.png')
    plt.show()