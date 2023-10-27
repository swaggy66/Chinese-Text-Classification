![image](https://github.com/swaggy66/Chinese-Text-Classification/assets/91064816/256d31f4-03b9-43a2-8ec8-c295d8de954b)![image](https://github.com/swaggy66/Chinese-Text-Classification/assets/91064816/3726db27-b55b-441b-817e-432ffead7c61)![image](https://github.com/swaggy66/Chinese-Text-Classification/assets/91064816/f96a0658-2cd4-4440-890b-2f2e3f74834d)![image](https://github.com/swaggy66/Chinese-Text-Classification/assets/91064816/0c88f504-3b7a-4dea-99d4-2616a4abe258)![image](https://github.com/swaggy66/Chinese-Text-Classification/assets/91064816/996aea44-8185-4ac9-8f8d-83887f48669b)# Chinese-Text-Classification
medical text classification

# A Model Ensemble Approach With LLM for Chinese Text Classification

![image](https://github.com/swaggy66/Chinese-Text-Classification/assets/91064816/b6504dff-11d3-4d03-b87c-549bc1fcb2b2)

# Model Selection

Qwen-7b-Chat，ChatGLM2-6b，Macbert

# Train methods

Qlora ，lora，FGM adversarial train

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

<img width="266" alt="image" src="https://github.com/swaggy66/Chinese-Text-Classification/assets/91064816/9abc3923-a355-4d74-9cf5-f15c836613d8">

# Run
1.run data_process.py
2.run sh lora.sh or train.py
3.run batch.py
4.run data_postprocess.py
