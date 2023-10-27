import json

# 定义选项映射字典
options_mapping = {
    '0': 'A.诊断',
    '1': 'B.治疗',
    '2': 'C.医疗常识',
    '3': 'D.健康生活方式',
    '4': 'E.流行病学',
    '5': 'F.其他'
}

# 读取txt文件
with open('/home/wuchengyan2/Firefly-master/medical_classify/pred_13911.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 处理每一行的内容并保存为JSONL格式
with open('/home/wuchengyan2/Firefly-master/medical_classify/process_data/pred1013.jsonl', 'w', encoding='utf-8') as jsonl_file:
    for line in lines:
        line_parts = line.strip().split('\t')
        if len(line_parts) == 2:
            text, option = line_parts
            option = option.strip()
            if option in options_mapping:
                conversation = {
                    "human": f"{text}? 请对上述的句子划分类别，从下面提供的类别中选出一个正确的选项：A.诊断 B.治疗 C.医疗常识 D.健康生活方式 E.流行病学 F.其他",
                    "assistant": options_mapping[option]
                }
                jsonl_file.write(json.dumps({"conversation": [conversation]}, ensure_ascii=False) + '\n')

print("处理完成，已保存到output.jsonl文件中。")
