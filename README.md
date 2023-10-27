# Chinese-Text-Classification
Chinese Text Classification With MacBERT， with simply 5 files

## 简介

本项目基于pytorch，利用MacBERT实现的中文文本分类，仅包含5个主要代码文件

本项目使用的数据集为csv格式的医疗诊断数据集，accuracy为top-k评价指标

项目的目录如下：

├─pretrained
		
├─project_data

├─result

│  ├─logs

│  │  └─medical-01

│  └─models

│      └─medical-01



其中`pretrained`部分需要自己下载，`logs`、`models`会自行生成

## 使用

部分数据，包括预训练模型，可以从[hugging_face](https://huggingface.co/hfl/chinese-macbert-base/tree/main) 下载，下载所有`json`文件、`vocab.txt`文件、与你需要的预训练模型参数文件（本文中即`pytorch_model.bin`）放在同一个目录（本项目为pretrained)

使用前可以先运行`dataset.py`（稍微调整main的部分）生成并保存分词后的数据

运行`train.py`开始训练

运行`test.py`进行测试



调整一些参数统一在`train.py`里的Config里面修改
