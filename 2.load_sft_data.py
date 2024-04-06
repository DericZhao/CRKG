import json

prompt_dict = {}

modes = ['train', 'valid', 'test']
data = []
for mode in modes:
    prompt1_file = 'sft/' + mode + '.json'

    if mode != 'test':
        with open(prompt1_file, 'r', encoding='utf-8') as file1:
            for line1 in file1.readlines():
                data.append(json.loads(line1))


prompt_file = 'sft/sft2.json'

with open(prompt_file, "w") as f:
    for line in data:
        if len(line["conversation"]) != 0:
            json.dump(line, f)
            f.write('\n')
    print("加载入文件完成...")
