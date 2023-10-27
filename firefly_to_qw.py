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

# # 读取txt文件
# with open('./class025transfer_label.txt', 'r', encoding='utf-8') as file:
#     lines = file.readlines()
#
# # 处理每一行的内容并保存为JSON格式
# output_data = []

input_file = '/data/wuchengyan/Qwen-main/data/process_data_qw/twostage.jsonl'
input_file1 = '/data/wuchengyan/Qwen-main/data/process_data_firefly/dev0926.jsonl'
output_file = '/data/wuchengyan/Qwen-main/data/process_data_firefly/dev0926.jsonl'

output_data = []
# 打开输入文件和输出文件
true_number = 0
all_number = 0
all_true =[]
with open(input_file, 'r', encoding='utf-8') as f_in, open(input_file1, 'r', encoding='utf-8') as f_in1, open(
        output_file, 'w', encoding='utf-8') as f_out:
    for line in (f_in):
        # 将每一行解析为JSON对象
        all_number += 1
        data = json.loads(line)
        # text, option = line_parts
        text = data['conversation'][0]["human"]
        option = data['conversation'][0]["assistant"]
        # option = option.strip()
        # if option in options_mapping:
        conversation_user = {
            "from": "user",
            "value": text
        }
        conversation_assistant = {
            "from": "assistant",
            "value": option
        }
        json_data = {
            "id": "id_0",
            "conversations": [conversation_user, conversation_assistant]
        }
        # print(json_data)
        # asd
        output_data.append(json_data)
#双阶段
# 保存为JSON格式
with open('/data/wuchengyan/Qwen-main/data/process_data_qw/v1adddata/correcttest1016two.json', 'w', encoding='utf-8') as json_file:
    json.dump(output_data, json_file, ensure_ascii=False, indent=2)

print("处理完成，已保存到output.json文件中。")
