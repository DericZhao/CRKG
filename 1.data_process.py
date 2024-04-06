import csv
import json

mode = 'valid'

data_file = f'ubuntu_v2/{mode}.csv'
target_file = f'sft/{mode}.json'

prompts = []
with open(data_file, 'r') as f:
    reader = csv.reader(f)
    reader.__next__()
    if mode == 'train':
        for id, row in enumerate(reader):
            # print(row)
            prompt_dict = {}
            sentence_dict = {}
            if row[2] == "1.0":
                prompt = ('Predict the next sentence based on the Context. '
                          'Context: ' + row[0])

                sentence_dict["human"] = prompt
                sentence_dict["assistant"] = row[1]

                prompt_dict["conversation_id"] = int(id)
                prompt_dict["category"] = "Brainstorming"
                prompt_dict["conversation"] = [sentence_dict]
                prompt_dict["dataset"] = "ubuntu_" + mode
                prompts.append(prompt_dict)
    else:
        for id, row in enumerate(reader):
            prompt_dict = {}
            sentence_dict = {}

            prompt = ('Predict the next sentence based on the Context. '
                      'Context: ' + row[0])

            sentence_dict["human"] = prompt
            sentence_dict["assistant"] = row[1]

            prompt_dict["conversation_id"] = int(id)
            prompt_dict["category"] = "Brainstorming"
            prompt_dict["conversation"] = [sentence_dict]
            prompt_dict["dataset"] = "ubuntu_" + mode
            prompts.append(prompt_dict)



with open(target_file, "w") as f:
    for line in prompts:
        json.dump(line, f)
        f.write('\n')
    print("加载入文件完成...")



