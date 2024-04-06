import pickle
from tqdm import tqdm
import csv
import json


prompt_file = 'sft/test.json'

prompts = []
targets = []
with open(prompt_file, 'r') as f:
    for line1 in f.readlines():
        prompt = ('Predict the next sentence based on the Context. '
                  'Context: ' + json.loads(line1)['conversation'][0]['human'])

        prompts.append(prompt)
        targets.append(json.loads(line1)['conversation'][0]['assistant'])


from transformers import AutoTokenizer, LlamaForCausalLM
import platform

sys = platform.system()

if sys == "Windows":
    model = LlamaForCausalLM.from_pretrained('D:\Project\LLM\Llama-2-7b-hf', device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained('D:\Project\LLM\Llama-2-7b-hf')
    max_length = 200
elif sys == "Linux":
    model = LlamaForCausalLM.from_pretrained('/home/deric/Deric/Project/LLM/Llama-2-7b-hf', device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained('/home/deric/Deric/Project/LLM/Llama-2-7b-hf')
    max_length = 1024

test_text = []
for prompt in tqdm(prompts):
    inputs = tokenizer(prompt, return_tensors="pt")

    generate_ids = model.generate(inputs.input_ids.to('cuda'), max_length=max_length, do_sample=False, num_beams=2)
    text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    test_text.append([text])

save_file = './results/HRGsft1.csv'
with open(save_file, 'w', encoding='utf-8', newline='') as targetFile:
    fp_write = csv.writer(targetFile)
    for test, target in zip(test_text, targets):
        fp_write.writerow([test, target])





