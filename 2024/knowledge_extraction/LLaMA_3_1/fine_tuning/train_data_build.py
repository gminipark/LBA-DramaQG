import os
import json
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
tqdm.pandas()  # Enable the tqdm progress bar for pandas
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.model_selection import train_test_split
import random
random.seed(42)

from data_utils import plot_token_count_distribution, split


base_model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PAD_TOKEN = "<|pad|>"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")  # Compressed 데이터로 학습할 땐 이걸 빼야할 것 같은데???
]
tokenizer.add_special_tokens({"pad_token":PAD_TOKEN})  # pad token으로 eos_token 사용하는 대신 추가해줌
tokenizer.padding_side = "right"  # left -> right 수정함

# Train Set + Validation Set 모두 사용해서 학습용 데이터셋 구축
dataset_paths=["./Data/DramaQA_KG_Processed/KG_GOLD_TRAIN.json", "./Data/DramaQA_KG_Processed/KG_GOLD_VAL.json"]
schema_paths=["./Data/DramaQA_KG_Processed/KG_SCHEMA_TRAIN.json", "./Data/DramaQA_KG_Processed/KG_SCHEMA_VAL.json", "./Data/DramaQA_KG_Processed/KG_SCHEMA_TEST.json"]
# schema_paths=["./Data/DramaQA_KG_Processed/KG_SCHEMA_TEST.json"]

# 학습용 데이터 로드
dataset = []
for dataset_path in dataset_paths:
    with open(dataset_path, 'r') as f:
        data = json.load(f)  # GOLD 데이터
        dataset.extend(data)
        
# 스키마 로드
schemas = {"objects":[], "attributes":[], "relations":[]}
for schema_path in schema_paths:
    with open(schema_path, 'r') as f:
        data_dict = json.load(f)  # 스키마
        schemas["objects"].extend(data_dict["objects"])
        schemas["attributes"].extend(data_dict["attributes"])
        schemas["relations"].extend(data_dict["relations"])
# 스키마 중복제거
schemas["objects"] = list(set(schemas["objects"]))
schemas["attributes"] = list(set(schemas["attributes"]))
schemas["relations"] = list(set(schemas["relations"]))

print(f'# of Objects in Schema:{len(schemas["objects"])} (Before Sampling)')
print(f'# of Attributes in Schema:{len(schemas["attributes"])} (Before Sampling)')
print(f'# of Relations in Schema:{len(schemas["relations"])} (Before Sampling)')

# 스키마 샘플링 (토큰카운트 줄이기 위해 10%만 남기도록 랜덤샘플링)
print(f"Randomly Sampling 10% of Schemas")

schemas["objects"] = random.sample(schemas["objects"], int(len(schemas["objects"]) * 0.1))
schemas["attributes"] = random.sample(schemas["attributes"], int(len(schemas["attributes"]) * 0.1))
schemas["relations"] = random.sample(schemas["relations"], int(len(schemas["relations"]) * 0.1))

print(f'# of Objects in Schema:{len(schemas["objects"])} (After Sampling)')
print(f'# of Attributes in Schema:{len(schemas["attributes"])} (After Sampling)')
print(f'# of Relations in Schema:{len(schemas["relations"])} (After Sampling)')

print("saving Sampled Schema to JSON")
with open("Data/DramaQA_KG_Processed/KG_SCHEMA_SAMPLED.json", 'w') as json_file:
    json.dump(schemas, json_file, indent=4)
print("saved")

fewShot_demo_path = "./Data/FewShotDemo/single_turn_demo.json"
# 퓨샷 예시 로드
with open(fewShot_demo_path, 'r') as f:
    demonstrations = json.load(f)

df = pd.DataFrame(dataset)
# print(df.head(3))

final_format_description = """
    Your responses must strictly adhere to the JSON formats provided below.
    Always avoid including any additional notes, explanations, or extra text outside the JSON structure.
    Make sure to find all triples and doubles you are confident about, but if you are not confident it's okay not to extract either one of them.

    Respond with the following JSON string format: 
    {
        "extracted_KGs": [
            {
                "object": "",
                "attribute": ""
            },
            {
                "object": "",
                "attribute": ""
            },
            {
                "object1": "",
                "relation": "",
                "object2": ""
            },
            {
                "object1": "",
                "relation": "",
                "object2": ""
            }
        ]
    }
    If a double or triple is incomplete or cannot be determined, do not include it in the response.
    The JSON must be valid and contain only the required fields.
    Make sure not to contain trailing comma at the last element in each lists.
    Make sure you have all the parenthesis, braces, and brackets correct so that json structure is preserved.
    If there's nothing to extract, just leave the value for the key "extracted_KGs" as empty list like such: [].
    """
system_message_schema_description = f"""\nFollowings are objects, attributes, and relations schemas defined in the current Knowledge-Base.
    objects: {schemas['objects']}
    attributes: {schemas['attributes']}
    relations: {schemas['relations']}
    Make sure you extract objects, attributes, and relations that already exist in the provided schemas.
    """
system_message = """
    You are an Knowledge Graph (KG) Extraction system designed to extract objects, relations and attributes from the given text, and create triples or doubles with them.
    Ensure all triples and doubles are complete and accurately reflect the information in the input text.""" + "\n" + final_format_description + "\n" + system_message_schema_description



def format_Data(row: dict, promptType: str):
    if promptType == "ZeroShot":
        final_prompt=[
            {"role": "user", "content": f"Extract list of KG(Knowledge Graph) doubles and triples from the following sentence: {row['sentence']}"},
            {"role": "assistant", "content": row['gold']}]  # 정답에 해당하는 추출 결과까지 포함
        formatted_text = tokenizer.apply_chat_template([{"role": "system", "content": system_message}] + final_prompt, tokenize=False, add_generation_prompt=False)  # add_generation_prompt는 False로 설정해야함!!! (입력한 프롬프트 체인 마지막에 <assistant> 태그를 넣어주는 기능이므로 학습 데이터 구축하는 본 과정에선 없는게 맞음)
    elif promptType == "OneShot":
        fs_prompt = []  # Initialize the prompt list with few-shot demonstrations
        fs_prompt.append({"role": "user", "content": f"Extract list of KG(Knowledge Graph) doubles and triples from the following sentence: {demonstrations[0]['input']}"})
        fs_prompt.append({"role": "assistant", "content": json.dumps({"extracted_KGs": demonstrations[0]["output"]["gold"]}, indent=4)})
        final_prompt=[
            {"role": "user", "content": f"Extract list of KG(Knowledge Graph) doubles and triples from the following sentence: {row['sentence']}"},
            {"role": "assistant", "content": row["gold"]}]  # 정답에 해당하는 추출 결과까지 포함
        formatted_text = tokenizer.apply_chat_template([{"role": "system", "content": system_message}] + fs_prompt + final_prompt, tokenize=False, add_generation_prompt=False)  # add_generation_prompt는 False로 설정해야함!!! (입력한 프롬프트 체인 마지막에 <assistant> 태그를 넣어주는 기능이므로 학습 데이터 구축하는 본 과정에선 없는게 맞음)
    elif promptType == "FewShot":
        fs_prompt = []  # Initialize the prompt list with few-shot demonstrations
        for demonstration in demonstrations:  # Few-Shot Demonstrations (added before the target sentence)
            fs_prompt.append({"role": "user", "content": f"Extract list of KG(Knowledge Graph) doubles and triples from the following sentence: {demonstration['input']}"})
            fs_prompt.append({"role": "assistant", "content": json.dumps({"extracted_KGs": demonstration["output"]["gold"]}, indent=4)})
        final_prompt=[
            {"role": "user", "content": f"Extract list of KG(Knowledge Graph) doubles and triples from the following sentence: {row['sentence']}"},
            {"role": "assistant", "content": row["gold"]}]  # 정답에 해당하는 추출 결과까지 포함
        formatted_text = tokenizer.apply_chat_template([{"role": "system", "content": system_message}] + fs_prompt + final_prompt, tokenize=False, add_generation_prompt=False)  # add_generation_prompt는 False로 설정해야함!!! (입력한 프롬프트 체인 마지막에 <assistant> 태그를 넣어주는 기능이므로 학습 데이터 구축하는 본 과정에선 없는게 맞음)
    else:
        print('error_wrong prompt type definition')
    
    return formatted_text

def format_Data_Compressed(rows: list[dict], promptType: str):
    if promptType == "ZeroShot":
        final_prompt = []
        for row in rows:
            final_prompt.append({"role": "user", "content": f"Extract list of KG(Knowledge Graph) doubles and triples from the following sentence: {row['sentence']}"})
            final_prompt.append({"role": "assistant", "content": row["gold"]})  # 정답에 해당하는 추출 결과까지 포함
        formatted_text = tokenizer.apply_chat_template([{"role": "system", "content": system_message}] + final_prompt, tokenize=False, add_generation_prompt=False)  # add_generation_prompt는 False로 설정해야함!!! (입력한 프롬프트 체인 마지막에 <assistant> 태그를 넣어주는 기능이므로 학습 데이터 구축하는 본 과정에선 없는게 맞음)
    elif promptType == "OneShot":
        fs_prompt = []  # Initialize the prompt list with few-shot demonstrations
        fs_prompt.append({"role": "user", "content": f"Extract list of KG(Knowledge Graph) doubles and triples from the following sentence: {demonstrations[0]['input']}"})
        fs_prompt.append({"role": "assistant", "content": json.dumps({"extracted_KGs": demonstrations[0]["output"]["gold"]}, indent=4)})
        final_prompt = []
        for row in rows:
            final_prompt.append({"role": "user", "content": f"Extract list of KG(Knowledge Graph) doubles and triples from the following sentence: {row['sentence']}"},)
            final_prompt.append({"role": "assistant", "content": row["gold"]})  # 정답에 해당하는 추출 결과까지 포함
        formatted_text = tokenizer.apply_chat_template([{"role": "system", "content": system_message}] + fs_prompt + final_prompt, tokenize=False, add_generation_prompt=False)  # add_generation_prompt는 False로 설정해야함!!! (입력한 프롬프트 체인 마지막에 <assistant> 태그를 넣어주는 기능이므로 학습 데이터 구축하는 본 과정에선 없는게 맞음)
    elif promptType == "FewShot":
        fs_prompt = []  # Initialize the prompt list with few-shot demonstrations
        for demonstration in demonstrations:  # Few-Shot Demonstrations (added before the target sentence)
            fs_prompt.append({"role": "user", "content": f"Extract list of KG(Knowledge Graph) doubles and triples from the following sentence: {demonstration['input']}"})
            fs_prompt.append({"role": "assistant", "content": json.dumps({"extracted_KGs": demonstration["output"]["gold"]}, indent=4)})
        final_prompt = []
        for row in rows:
            final_prompt.append({"role": "user", "content": f"Extract list of KG(Knowledge Graph) doubles and triples from the following sentence: {row['sentence']}"})
            final_prompt.append({"role": "assistant", "content": row["gold"]})  # 정답에 해당하는 추출 결과까지 포함
        formatted_text = tokenizer.apply_chat_template([{"role": "system", "content": system_message}] + fs_prompt + final_prompt, tokenize=False, add_generation_prompt=False)  # add_generation_prompt는 False로 설정해야함!!! (입력한 프롬프트 체인 마지막에 <assistant> 태그를 넣어주는 기능이므로 학습 데이터 구축하는 본 과정에선 없는게 맞음)
    else:
        print('error_wrong prompt type definition')
    
    return formatted_text

def count_Tokens(row: dict, column_name: str) -> int:
    return len(
        tokenizer(
            row[column_name],
            add_special_tokens=True,
            return_attention_mask=False,
        )["input_ids"]
    )

print("Processing: TrainData-Text Formatting (Uncompressed) (Schema + Demos + Q-A)")
# Schema + Demos + Q-A
tqdm.pandas(desc="Processing: Formatting (ZeroShot)"); df["Text_ZeroShot"] = df.progress_apply(lambda row: format_Data(row, "ZeroShot"), axis=1)
tqdm.pandas(desc="Processing: Tokenizing/Counting (ZeroShot)"); df["Token_Count_ZeroShot"] = df.progress_apply(lambda row: count_Tokens(row, "Text_ZeroShot"), axis=1)
tqdm.pandas(desc="Processing: Formatting (OneShot)"); df["Text_OneShot"] = df.progress_apply(lambda row: format_Data(row, "OneShot"), axis=1)
tqdm.pandas(desc="Processing: Tokenizing/Counting (OneShot)"); df["Token_Count_OneShot"] = df.progress_apply(lambda row: count_Tokens(row, "Text_OneShot"), axis=1)
tqdm.pandas(desc="Processing: Formatting (FewShot)"); df["Text_FewShot"] = df.progress_apply(lambda row: format_Data(row, "FewShot"), axis=1)
tqdm.pandas(desc="Processing: Tokenizing/Counting (FewShot)"); df["Token_Count_FewShot"] = df.progress_apply(lambda row: count_Tokens(row, "Text_FewShot"), axis=1)


print("Processing: TrainData-Text Formatting (Compressed) (Schema + Demos + N(Q-A))")
# Create an empty list to store results
compressed_Text_ZeroShot = []
compressed_Text_OneShot = []
compressed_Text_FewShot = []
# Set batch size
batch_size = 10
for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):  # Iterate over the DataFrame in batches of 10
    batch = df.iloc[i:i + batch_size].to_dict(orient='records')
    # print(batch)
    batch_results = format_Data_Compressed(batch, "ZeroShot")
    compressed_Text_ZeroShot.append(batch_results)
    batch_results = format_Data_Compressed(batch, "OneShot")
    compressed_Text_OneShot.append(batch_results)
    batch_results = format_Data_Compressed(batch, "FewShot")
    compressed_Text_FewShot.append(batch_results)
# print(compressed_Text_FewShot[0:10])

# Schema + Demos + #(Q-A)
df_compressed = pd.DataFrame()
tqdm.pandas(desc="Processing: Formatting (ZeroShot)"); df_compressed["Text_ZeroShot_Compressed"] = compressed_Text_ZeroShot
tqdm.pandas(desc="Processing: Tokenizing/Counting (ZeroShot)"); df_compressed["Token_Count_ZeroShot_Compressed"] = df_compressed.progress_apply(lambda row: count_Tokens(row, "Text_ZeroShot_Compressed"), axis=1)
tqdm.pandas(desc="Processing: Formatting (OneShot)"); df_compressed["Text_OneShot_Compressed"] = compressed_Text_OneShot
tqdm.pandas(desc="Processing: Tokenizing/Counting (OneShot)"); df_compressed["Token_Count_OneShot_Compressed"] = df_compressed.progress_apply(lambda row: count_Tokens(row, "Text_OneShot_Compressed"), axis=1)
tqdm.pandas(desc="Processing: Formatting (FewShot)"); df_compressed["Text_FewShot_Compressed"] = compressed_Text_FewShot
tqdm.pandas(desc="Processing: Tokenizing/Counting (FewShot)"); df_compressed["Token_Count_FewShot_Compressed"] = df_compressed.progress_apply(lambda row: count_Tokens(row, "Text_FewShot_Compressed"), axis=1)


# Print to Check the Resulting DataFrame
# print(df.head())
# print(df_compressed.head())

# Save Plots
plot_token_count_distribution(df.Token_Count_ZeroShot)
plot_token_count_distribution(df.Token_Count_OneShot)
plot_token_count_distribution(df.Token_Count_FewShot)
plot_token_count_distribution(df_compressed.Token_Count_ZeroShot_Compressed)
plot_token_count_distribution(df_compressed.Token_Count_OneShot_Compressed)
plot_token_count_distribution(df_compressed.Token_Count_FewShot_Compressed)

train_uncomp, val_uncomp, test_uncomp = split(df)
train_comp, val_comp, test_comp = split(df_compressed)

# Print Length/Ratio of Splits
print(f'Ratio: Train({len(train_comp)/len(df_compressed)}) / Val({len(val_comp)/len(df_compressed)}) / Test({len(test_comp)/len(df_compressed)})')
print(f'Count: Train({len(train_comp)}) / Val({len(val_comp)}) / Test({len(test_comp)})')
print(f'Shape: Train({train_comp.shape}) / Val({val_comp.shape}) / Test({test_comp.shape})')

# print(train_comp.loc[0, "Text_FewShot_Compressed"])
# print(val_comp.loc[0, "Text_FewShot_Compressed"])
# print(test_comp.loc[0, "Text_FewShot_Compressed"])

# Save Splits
print("saving...")

# UnCompressed Dataset Directories
uncompressed_dir = "./Data/DramaQA_KG_Processed/TrainingData/uncompressed/"
train_uncomp_dir = os.path.join(uncompressed_dir, "train_uncomp.json")
val_uncomp_dir = os.path.join(uncompressed_dir, "val_uncomp.json")
test_uncomp_dir = os.path.join(uncompressed_dir, "test_uncomp.json")
train_uncomp_zeroshot_dir = os.path.join(uncompressed_dir, "train_uncomp_zeroshot.json")
train_uncomp_oneshot_dir = os.path.join(uncompressed_dir, "train_uncomp_oneshot.json")
train_uncomp_fewshot_dir = os.path.join(uncompressed_dir, "train_uncomp_fewshot.json")
val_uncomp_zeroshot_dir = os.path.join(uncompressed_dir, "val_uncomp_zeroshot.json")
val_uncomp_oneshot_dir = os.path.join(uncompressed_dir, "val_uncomp_oneshot.json")
val_uncomp_fewshot_dir = os.path.join(uncompressed_dir, "val_uncomp_fewshot.json")
test_uncomp_zeroshot_dir = os.path.join(uncompressed_dir, "test_uncomp_zeroshot.json")
test_uncomp_oneshot_dir = os.path.join(uncompressed_dir, "test_uncomp_oneshot.json")
test_uncomp_fewshot_dir = os.path.join(uncompressed_dir, "test_uncomp_fewshot.json")

# Compressed Dataset Directories
compressed_dir = "./Data/DramaQA_KG_Processed/TrainingData/compressed/"
train_comp_dir = os.path.join(compressed_dir, "train_comp.json")
val_comp_dir = os.path.join(compressed_dir, "val_comp.json")
test_comp_dir = os.path.join(compressed_dir, "test_comp.json")
train_comp_zeroshot_dir = os.path.join(compressed_dir, "train_comp_zeroshot.json")
train_comp_oneshot_dir = os.path.join(compressed_dir, "train_comp_oneshot.json")
train_comp_fewshot_dir = os.path.join(compressed_dir, "train_comp_fewshot.json")
val_comp_zeroshot_dir = os.path.join(compressed_dir, "val_comp_zeroshot.json")
val_comp_oneshot_dir = os.path.join(compressed_dir, "val_comp_oneshot.json")
val_comp_fewshot_dir = os.path.join(compressed_dir, "val_comp_fewshot.json")
test_comp_zeroshot_dir = os.path.join(compressed_dir, "test_comp_zeroshot.json")
test_comp_oneshot_dir = os.path.join(compressed_dir, "test_comp_oneshot.json")
test_comp_fewshot_dir = os.path.join(compressed_dir, "test_comp_fewshot.json")

# Saving UnCompressed Dataset
train_uncomp.to_json(train_uncomp_dir, orient="records", lines=True)
val_uncomp.to_json(val_uncomp_dir, orient="records", lines=True)
test_uncomp.to_json(test_uncomp_dir, orient="records", lines=True)

train_uncomp[["Text_ZeroShot"]].to_json(train_uncomp_zeroshot_dir, orient="records", lines=True)
train_uncomp[["Text_OneShot"]].to_json(train_uncomp_oneshot_dir, orient="records", lines=True)
train_uncomp[["Text_FewShot"]].to_json(train_uncomp_fewshot_dir, orient="records", lines=True)
val_uncomp[["Text_ZeroShot"]].to_json(val_uncomp_zeroshot_dir, orient="records", lines=True)
val_uncomp[["Text_OneShot"]].to_json(val_uncomp_oneshot_dir, orient="records", lines=True)
val_uncomp[["Text_FewShot"]].to_json(val_uncomp_fewshot_dir, orient="records", lines=True)
test_uncomp[["Text_ZeroShot"]].to_json(test_uncomp_zeroshot_dir, orient="records", lines=True)
test_uncomp[["Text_OneShot"]].to_json(test_uncomp_oneshot_dir, orient="records", lines=True)
test_uncomp[["Text_FewShot"]].to_json(test_uncomp_fewshot_dir, orient="records", lines=True)

# Saving Compressed Dataset
train_comp.to_json(train_comp_dir, orient="records", lines=True)
val_comp.to_json(val_comp_dir, orient="records", lines=True)
test_comp.to_json(test_comp_dir, orient="records", lines=True)

train_comp[["Text_ZeroShot_Compressed"]].to_json(train_comp_zeroshot_dir, orient="records", lines=True)
train_comp[["Text_OneShot_Compressed"]].to_json(train_comp_oneshot_dir, orient="records", lines=True)
train_comp[["Text_FewShot_Compressed"]].to_json(train_comp_fewshot_dir, orient="records", lines=True)
val_comp[["Text_ZeroShot_Compressed"]].to_json(val_comp_zeroshot_dir, orient="records", lines=True)
val_comp[["Text_OneShot_Compressed"]].to_json(val_comp_oneshot_dir, orient="records", lines=True)
val_comp[["Text_FewShot_Compressed"]].to_json(val_comp_fewshot_dir, orient="records", lines=True)
test_comp[["Text_ZeroShot_Compressed"]].to_json(test_comp_zeroshot_dir, orient="records", lines=True)
test_comp[["Text_OneShot_Compressed"]].to_json(test_comp_oneshot_dir, orient="records", lines=True)
test_comp[["Text_FewShot_Compressed"]].to_json(test_comp_fewshot_dir, orient="records", lines=True)

print("saved")

# Load_Dataset
print("Loading Dataset")
dataset_compressed = load_dataset(
    "json",
    data_files={"train": train_comp_dir, "validation": val_comp_dir, "test": test_comp_dir},
)
print("Done")
print(dataset_compressed)