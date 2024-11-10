
import os
import json
from tqdm import tqdm
from datasets import load_dataset
import re

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import LlamaTokenizerFast, MistralForCausalLM
from huggingface_hub import snapshot_download
from transformers import BitsAndBytesConfig


with open("QA/DramaQA/AnotherMissOhQA_val_set.json", "r") as f:
    train_dataset = json.load(f)
 

model_name = "Mistral-Nemo-Instruct-2407"
cache_dir = model_name.split('/')[-1] if '/' in model_name else model_name
token = ''
snapshot_download(repo_id=model_name,  token=token, cache_dir=cache_dir,
                  ignore_patterns='consolidated.safetensors',
                  local_dir=cache_dir, max_workers=1)

quantization_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=cache_dir,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    token=token,
    cache_dir=cache_dir,
     quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(cache_dir, 
                                          token=token, 
                                          trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token
model.generation_config.pad_token_id = tokenizer.pad_token_id

start_idx = 0
end_idx = len(train_dataset)
for idx, example in tqdm(enumerate(train_dataset[start_idx:end_idx]), total=len(range(start_idx, end_idx))):

    messages = [
    {"role": "system", 
    "content": ('You are a helpful assistant.'
                'Extract entity, relation from the english sentence about video.'
                'output template is json that is \"\n{\"source entity\": \"{noun}\", \"relation\": \"{verb}\", \"target entity\": \"{noun}"}\"'
                'example:'
                '\n### input: \n'
                'Because Deogi had to call Dokyung.'
                '\n### ouput: \n'
                '{"source entity": "Deogi",'
                '"relation": "call",'
                '"target entity": "Dokyung"}'
                '\n### input: \n'
                'Kyungsu is Deogi\'s high school friend and Kyungsu hates Deogi.'
                '\n### output: \n'
                '[{"source entity": "Kyungsu",'
                '"relation": "friend of",'
                '"target entity": "Deogi"},'
                '{"source entity": "Kyungsu",'
                '"relation": "hate",'
                '"target entity": "Deogi"}]'
                '\n### input: \n'
                'Deogi is Haeyoung1\'s middle school teacher.'
                '\n### output: \n'
                '{"source entity": "Deogi",'
                '"relation": "teacher of",'
                '"target entity": "Haeyong1"}'
                '\n### input: \n'
                'Deogi eats food a lot.'
                '\n### output: \n'
                '{"source entity": "Deogi",'
                '"relation": "eat",'
                '"target entity": "food"}')
    },
    {"role": "user", "content": (
                            "\n### input: \n"
                            f"{example['answers'][example['correct_idx']]}"
                            "\n### output: \n")}
    ]
    
    input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

    output = model.generate(
        input_ids.to("cuda"),
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=256,
    )

    result = tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens = True)
    # print('---------------------')
    # print(example['answers'][example['correct_idx']])
    # print(result)
    try:
        if "output:" in result:
            result = result.split("output:")[-1].strip()

        print(result)
        new_result = json.loads(result)
        if isinstance(new_result, dict):
            new_result = [new_result]
    except :
        new_result = []
    # print('-----------------------------------')
    # print(new_result)
    example['answer_meta'] = new_result
    
    
with open(f'QA/DramaQA/AnotherMissOhQA_val_set_v2_{start_idx}_{end_idx-1}.json', 'w') as f:
    json.dump(train_dataset, f)