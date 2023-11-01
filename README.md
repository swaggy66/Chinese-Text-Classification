medical text classification

# A Model Ensemble Approach With LLM for Chinese Text Classification

![总模型图初稿](https://github.com/swaggy66/Chinese-Text-Classification/assets/91064816/5d879b4f-9360-4558-93f6-1f9caa9fa333)


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

# rank（top2）

<img width="1186" alt="image" src="https://github.com/swaggy66/Chinese-Text-Classification/assets/91064816/cde1dad6-b85d-437b-809e-3362d8850745">


