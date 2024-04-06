import csv
import json

mode = 'test'

data_file = f'dailydialogue/dial.{mode}'
target_file = f'sft/{mode}_dial.json'

prompts = []
with open(data_file, 'r') as f:
    lines = f.readlines()

    for id, line in enumerate(lines):
        length = len(line.strip('\n').split('\t'))

        context = line.strip('\n').split('\t')[0]
        response = line.strip('\n').split('\t')[1]

        prompt_dict = {}
        sentence_dict = {}

        prompt = ('Predict the next sentence based on the Context. '
                  'Context: ' + context)

        sentence_dict["human"] = prompt
        sentence_dict["assistant"] = response

        prompt_dict["conversation_id"] = int(id)
        prompt_dict["category"] = "Brainstorming"
        prompt_dict["conversation"] = [sentence_dict]
        prompt_dict["dataset"] = "daily_dialogue" + mode
        prompts.append(prompt_dict)




with open(target_file, "w") as f:
    for line in prompts:
        json.dump(line, f)
        f.write('\n')
    print("加载入文件完成...")



