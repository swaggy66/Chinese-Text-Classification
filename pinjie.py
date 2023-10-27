import json

# 打开txt文件和jsonl文件
txt_file = open('/home/wuchengyan2/Firefly-master/medical_classify/raw_data/pred.txt', 'r')
jsonl_file = open('/home/wuchengyan2/Firefly-master/medical_classify/predit1010/pred1015.jsonl', 'r')

# 读取txt文件内容到txt_lines列表
txt_lines = txt_file.readlines()

# 打开一个新的txt文件用于写入转换后的内容 
output_file = open('/home/wuchengyan2/Firefly-master/medical_classify/predit1010/pred1015output.txt', 'w')

# 字典用于匹配jsonl中assistant的值和相应的数字
assistant_dict = {'A.诊断': '0', 'B.治疗': '1', 'C.医疗常识': '2', 'D.健康生活方式': '3', 'E.流行病学': '4', 'F.其他': '5'}

# 遍历jsonl文件每行
for line in jsonl_file:
    # loads每行json格式数据
    data = json.loads(line)
    assistant = data['conversation'][0]['assistant']
    # 获取assistant的值对应的数字
    num = assistant_dict[assistant]
    
    # 写入原txt内容
    output_file.write(txt_lines.pop(0).rstrip('\n'))  # 去除原始行的换行符
    # 添加分隔符和assistant数字
    output_file.write('\t'+num)
    # 每行结束后添加换行符
    output_file.write('\n')

# 关闭所有文件
txt_file.close() 
jsonl_file.close()
output_file.close()
