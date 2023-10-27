# Chinese-Text-Classification
medical text classification

A Model Ensemble Approach With LLM for Chinese Text Classification

![image](https://github.com/swaggy66/Chinese-Text-Classification/assets/91064816/23d8917c-8112-44a9-b353-37aa6fd55268)

# data format
{
    "id": "id_0",
    "conversations": [
      {
        "from": "user",
        "value": "糖尿病人可不可以服用伟哥? 请对上述的句子划分类别，从下面提供的类别中选出一个正确的选项：A.诊断 B.治疗 C.医疗常识 D.健康生活方式 E.流行病学 F.其他"
      },
      {
        "from": "assistant",
        "value": "B.治疗"
      }
    ]
  }

# label transfer

<img width="266" alt="image" src="https://github.com/swaggy66/Chinese-Text-Classification/assets/91064816/052cb2df-8850-4eab-ac46-aa899253182d">

# Run
1.run data process.py
2.run sh lora.sh and train.py
3.run batch.py
4.run data postprocess.py
