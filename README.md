# 1.Topic：Medical Text Classification

contest and data link：https://tianchi.aliyun.com/competition/entrance/532153?spm=a2c22.12281957.0.0.4c885d9bzbNNxD

# 2.Paper Title：《A Model Ensemble Approach With LLM for Chinese Text Classification》

# 3.Architecture

<img width="879" alt="image" src="https://github.com/swaggy66/Chinese-Text-Classification/assets/91064816/40abb620-fd53-4140-a1c4-82aad6c73f4c">






# 4.Model Selection

Qwen-7b-Chat，ChatGLM2-6b，Macbert

# 5.Train methods

Qlora ，lora，FGM adversarial train

# 6.data format

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
# 7.label transfer

<img width="266" alt="image" src="https://github.com/swaggy66/Chinese-Text-Classification/assets/91064816/9abc3923-a355-4d74-9cf5-f15c836613d8">

# 8.Run
1.run data_process.py

2.run sh lora.sh or train.py

3.run batch.py

4.run data_postprocess.py

# 9.rank（top2）

<img width="1186" alt="image" src="https://github.com/swaggy66/Chinese-Text-Classification/assets/91064816/cde1dad6-b85d-437b-809e-3362d8850745">

# 10.Citation
<kbd style="background-color: #f2f2f2; padding: 10px; border-radius: 5px; display: block; text-align: center; font-size: 16px;">
    {email：18819893186wcy@gmail.com}
</kbd>


