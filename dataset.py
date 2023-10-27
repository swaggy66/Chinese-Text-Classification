#coding=gbk
from torch.utils.data import Dataset
import torch
import numpy as np
import csv
from tqdm import tqdm

PAD, CLS = '[PAD]', '[CLS]'

class MedicalDataset(Dataset):

    """用于打开已经分词后保存的结果文件，最后未用"""

    def __init__(self, config):
        self.data = np.load(config.path, allow_pickle=True)
        print(self.data[1])
        
    def __getitem__(self, idx):
        return [self.data[idx][0], self.data[idx][1]] # input , label
    
    def __len__(self):
        return len(self.data)

class MedicalDatasetRaw(Dataset):
    
    """用于打开原始的文件，并用Tokenizer分词"""
    
    def __init__(self, config, path, from_csv = True):
        
        self.class_dic = {}
        self.inputs = []
        self.labels = []
        self.path = path

        print("loading class names...")
        self.load_names()

        if from_csv:

            print("opening: " + path)
            total = sum(1 for _ in open(path, encoding='UTF8'))
            f = open(path, encoding='UTF8')
            f_csv = csv.reader(f)
            next(f_csv)

            bar = tqdm(f_csv, total=total)

            for line in bar:
                token = config.tokenizer.tokenize(line[0])
                token = [CLS] + token
                token_ids = config.tokenizer.convert_tokens_to_ids(token)
                label = self.get_label(line[1])
                
                if config.pad_size:
                    token_ids, mask, _ = self.pad(token_ids, config.pad_size)
                    self.inputs.append(torch.LongTensor([token_ids, mask]).to(config.device))
                    # self.inputs.append({'id':torch.LongTensor(token_ids).to(config.device), 'mask':torch.LongTensor(mask).to(config.device)})
                else:
                    self.inputs.append(torch.LongTensor([token_ids]).to(config.device))

                self.labels.append(torch.LongTensor([label]).to(config.device)) 

            f.close()

        else:
            print("opening: " + path + '-input.pt')
            self.inputs = torch.load(path + '-input.pt')

            print("opening: " + path + '-label.pt')
            self.labels = torch.load(path + '-label.pt')  

    def __getitem__(self, idx):
        return [self.inputs[idx], self.labels[idx]] # input , label
    
    def __len__(self):
        return len(self.inputs)

    def pad(self, ids, pad_size):

        """处理句子长度为固定长度"""

        token_ids = ids
        if len(token_ids) < pad_size:
            mask = [1] * len(token_ids) + [0] * (pad_size - len(token_ids))
            token_ids += ([0] * (pad_size - len(token_ids)))
            length = len(token_ids)
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
            length = pad_size
        
        return token_ids, mask, length

    def get_label(self, label):

        """根据字符串得到lebel的序号"""
        # print("label:", label, "order:",self.class_dic[label])
        return self.class_dic[label]

    def get_class_num(self):

        return len(self.class_dic)

    def load_names(self):
        
        """根据训练文件得到所有的类别名对应的序号"""

        f = open("project_data/train.csv", encoding='UTF8')
        f_csv = csv.reader(f)
        next(f_csv)
        idx = 0
        for line in f_csv:
            if line[1] not in self.class_dic:
                self.class_dic.setdefault(line[1], idx)
                idx = idx + 1

        f.close()

    def save(self):
        print("saving data...")
        torch.save(self.inputs, self.path[:-4] + '-input.pt')
        torch.save(self.labels, self.path[:-4] + '-label.pt')

if __name__ == "__main__":

    from train import Config
    opt = Config("medical-00")
    # t_d = MedicalDatasetRaw(opt, 'project_data/train.csv')
    # t_d.save()
    e_d = MedicalDatasetRaw(opt, 'project_data/test.csv')
    e_d.save()

    # t_d = MedicalDatasetRaw(opt, 'project_data/train', False)
    # print(t_d[1])
    # print(t_d[100])
