from pyexpat import model
import torch
import os
from tqdm import tqdm 
import torch.nn.functional as F
from dataset import MedicalDatasetRaw
from model import Model

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from train import Config,accuracy

if __name__ == "__main__":
    
    opt = Config("medical-test")
    opt.bert_path = "/data/wuchengyan/Chinese-Text-Classification-main/result/models/medical-02/4-0.1074742034887895-96.65.pt"

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
    embedding = []
    labels = []
    with torch.no_grad():
        for i, batch in bar:
            total_batch += 1
            inputs = batch[0]
            labels_batch = batch[1][:, 0].long()

            # Assuming the model returns embeddings and predictions
            model_outputs = model(inputs)

            # Print the shape of the model outputs to understand its structure
            print(f"Model Outputs Shape: {model_outputs.shape}")

            # Modify the unpacking based on the actual structure of model_outputs
            if isinstance(model_outputs, tuple) and len(model_outputs) == 2:
                x, outputs = model_outputs
            else:
                # If the model doesn't return a tuple, assume the first element is the embedding
                x = model_outputs
                outputs = model_outputs  # Modify this line if needed

            embedding.append(x)
            labels.append(labels_batch)

            loss = loss_f(outputs, labels_batch)  # not one hot
            acc = accuracy(outputs, labels_batch, topk=[3])[0]

            total_acc += acc.to('cpu').detach().numpy()[0]
            total_loss += loss.item()

            logger.add_scalar('loss', loss.item(), total_batch)
            logger.add_scalar('acc', acc.to('cpu').detach().numpy()[0], total_batch)

            probabilities = F.softmax(outputs, dim=1)
            for j in range(len(labels_batch)):
                print(
                    f"Sample {total_batch * opt.batch_size + j + 1}: Predicted Probabilities: {probabilities[j]}, Actual: {labels_batch[j]}")

        total_loss /= total_batch
        total_acc /= total_batch
        print(f'Train total Loss: {total_loss},  Train total Acc {total_acc}')


#####PCA可视化
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    X_std = StandardScaler().fit_transform(embedding)
    X_pca = PCA(n_components=2).fit_transform(X_std)
    X_pca = np.vstack((X_pca.T, labels)).T

    df_pca = pd.DataFrame(X_pca, columns=['1st_Component', '2n_Component', 'class'])
    df_pca.head()

    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df_pca, hue='class', x='1st_Component', y='2nd_Component')

    plt.savefig('./1.jpg')
    plt.show()

#####TSNE可视化
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(embedding)
    X_tsne_data = np.vstack((X_tsne.T, y)).T
    df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'class'])
    df_tsne.head()

    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df_tsne, hue='class', x='Dim1', y='Dim2')
    plt.savefig('./2.jpg')
    plt.show()