import sys
import torch
import numpy as np
import json
from LLaMA3_1.llama3_1_Tester_v6 import Tester_v6
from .compare import compareNplot
from test_utils import compute_acc, compute_score_for_sentence, classify_errors, eval_KG, test_case

sample_KGs = './Data/DramaQA_KG_Processed/KG_GOLD_TEST_500.json'
sample_sentences = [] # list of list of dicts
sample_gold_KGs = []  # list of list of dicts
with open(sample_KGs, 'r') as file:
    data = json.load(file)
data = data[:100]  # Test Purpose

for item in data:
    sentence = item['sentence'] # list of dicts
    gold_KG = item['gold']  # list of dicts

    # Append to respective lists
    sample_sentences.append(sentence)
    sample_gold_KGs.append(gold_KG)

schema_file = './Data/DramaQA_KG_Processed/KG_SCHEMA_TEST.json'
with open(schema_file, 'r') as file:
    schemas = json.load(file)

# Few-Shot Demonstrations - Single Turn
with open('./Data/FewShotDemo/single_turn_demo.json', 'r') as file:
    few_shot_demos_singleTurn = json.load(file)

# Few-Shot Demonstrations - Multi Turn QA
with open('./Data/FewShotDemo/multi_turn_demo.json', 'r') as file:
    few_shot_demos_multiTurnQA = json.load(file)

print('=========='*20); sys.stdout.flush()
print('=========='*20); sys.stdout.flush()


labels = [
            'Single-Turn : Zero-Shot + Schemas', 
            'Multi-Turn : Zero-Shot + Schemas', 
            'Single-Turn : One-Shot (1) + Schemas', 
            'Multi-Turn : One-Shot (1) + Schemas',
            'Single-Turn : Few-Shot (10) + Schemas',
            'Multi-Turn : Few-Shot (10) + Schemas'
        ]



""" LLaMA 3.1 8B [LoRA] - EPOCH 1 """
# Test Cases
precisions = []
recalls = []
f1s = []
all_error_cases = {}

model = 'LLaMA3.1-8B-LoRA-ZeroShot(Epoch1)'
lora_adapter_path = "./LLaMA3_1/fine_tuning/fine_tuned_models/BackUps/lora3.1_8B_rank32_alpha16_batch1_accumstep4_epoch5_ZeroShot/checkpoint-epoch1"

# 1) Single-Turn : Zero-Shot + Schemas
print(f'# 1) Single-Turn : Zero-Shot + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=None, schemas=schemas, label=labels[0], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[0]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 2) Multi-Turn : Zero-Shot + Schemas
print(f'# 2) Multi-Turn : Zero-Shot + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=None, schemas=schemas, isMultiTurn=True, label=labels[1], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[1]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 3) Single-Turn : One-Shot (1) + Schemas
print(f'# 3) Single-Turn : One-Shot (1) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_singleTurn[:1], schemas=schemas, label=labels[2], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[2]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 4) Multi-Turn : One-Shot (1) + Schemas
few_shot_demos_multiTurnQA = few_shot_demos_multiTurnQA[:1]  # Test용
# print(f'Multi-Turn Few-Shot Demos: {few_shot_demos_multiTurnQA}')  # Test용
print(f'# 4) Multi-Turn : One-Shot (1) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_multiTurnQA, schemas=schemas, isMultiTurn=True, label=labels[3], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[3]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 5) Single-Turn : Few-Shot (10) + Schemas'
print(f'# 5) Single-Turn : Few-Shot (10) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_singleTurn[:10], schemas=schemas, label=labels[4], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[4]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 6) Multi-Turn : Few-Shot (10) + Schemas
few_shot_demos_multiTurnQA = few_shot_demos_multiTurnQA[:10]  # Test용
# print(f'Multi-Turn Few-Shot Demos: {few_shot_demos_multiTurnQA}')  # Test용
print(f'# 4) Multi-Turn : Few-Shot (10) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_multiTurnQA, schemas=schemas, isMultiTurn=True, label=labels[5], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[5]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# PLOT
print('Saving Plot...'); sys.stdout.flush()
compareNplot(precisions, recalls, f1s, all_error_cases=all_error_cases, labels=labels, model=model)
print('Test Complete'); sys.stdout.flush()


########################################################################################################################################################################
print('=========='*20); sys.stdout.flush()
print('=========='*20); sys.stdout.flush()

""" LLaMA 3.1 8B [LoRA] - EPOCH 3 """
# Test Cases
precisions = []
recalls = []
f1s = []
all_error_cases = {}

model = 'LLaMA3.1-8B-LoRA-ZeroShot(Epoch3)'
lora_adapter_path = "./LLaMA3_1/fine_tuning/fine_tuned_models/BackUps/lora3.1_8B_rank32_alpha16_batch1_accumstep4_epoch5_ZeroShot/checkpoint-epoch3"

# 1) Single-Turn : Zero-Shot + Schemas
print(f'# 1) Single-Turn : Zero-Shot + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=None, schemas=schemas, label=labels[0], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[0]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 2) Multi-Turn : Zero-Shot + Schemas
print(f'# 2) Multi-Turn : Zero-Shot + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=None, schemas=schemas, isMultiTurn=True, label=labels[1], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[1]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 3) Single-Turn : One-Shot (1) + Schemas
print(f'# 3) Single-Turn : One-Shot (1) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_singleTurn[:1], schemas=schemas, label=labels[2], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[2]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 4) Multi-Turn : One-Shot (1) + Schemas
few_shot_demos_multiTurnQA = few_shot_demos_multiTurnQA[:1]  # Test용
# print(f'Multi-Turn Few-Shot Demos: {few_shot_demos_multiTurnQA}')  # Test용
print(f'# 4) Multi-Turn : One-Shot (1) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_multiTurnQA, schemas=schemas, isMultiTurn=True, label=labels[3], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[3]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 5) Single-Turn : Few-Shot (10) + Schemas'
print(f'# 5) Single-Turn : Few-Shot (10) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_singleTurn[:10], schemas=schemas, label=labels[4], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[4]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 6) Multi-Turn : Few-Shot (10) + Schemas
few_shot_demos_multiTurnQA = few_shot_demos_multiTurnQA[:10]  # Test용
# print(f'Multi-Turn Few-Shot Demos: {few_shot_demos_multiTurnQA}')  # Test용
print(f'# 4) Multi-Turn : Few-Shot (10) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_multiTurnQA, schemas=schemas, isMultiTurn=True, label=labels[5], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[5]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# PLOT
print('Saving Plot...'); sys.stdout.flush()
compareNplot(precisions, recalls, f1s, all_error_cases=all_error_cases, labels=labels, model=model)
print('Test Complete'); sys.stdout.flush()


########################################################################################################################################################################
print('=========='*20); sys.stdout.flush()
print('=========='*20); sys.stdout.flush()

""" LLaMA 3.1 8B [LoRA] - EPOCH 5 """
# Test Cases
precisions = []
recalls = []
f1s = []
all_error_cases = {}

model = 'LLaMA3.1-8B-LoRA-ZeroShot(Epoch5)'
lora_adapter_path = "./LLaMA3_1/fine_tuning/fine_tuned_models/BackUps/lora3.1_8B_rank32_alpha16_batch1_accumstep4_epoch5_ZeroShot/checkpoint-epoch5"

# 1) Single-Turn : Zero-Shot + Schemas
print(f'# 1) Single-Turn : Zero-Shot + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=None, schemas=schemas, label=labels[0], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[0]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 2) Multi-Turn : Zero-Shot + Schemas
print(f'# 2) Multi-Turn : Zero-Shot + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=None, schemas=schemas, isMultiTurn=True, label=labels[1], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[1]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 3) Single-Turn : One-Shot (1) + Schemas
print(f'# 3) Single-Turn : One-Shot (1) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_singleTurn[:1], schemas=schemas, label=labels[2], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[2]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 4) Multi-Turn : One-Shot (1) + Schemas
few_shot_demos_multiTurnQA = few_shot_demos_multiTurnQA[:1]  # Test용
# print(f'Multi-Turn Few-Shot Demos: {few_shot_demos_multiTurnQA}')  # Test용
print(f'# 4) Multi-Turn : One-Shot (1) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_multiTurnQA, schemas=schemas, isMultiTurn=True, label=labels[3], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[3]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 5) Single-Turn : Few-Shot (10) + Schemas'
print(f'# 5) Single-Turn : Few-Shot (10) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_singleTurn[:10], schemas=schemas, label=labels[4], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[4]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 6) Multi-Turn : Few-Shot (10) + Schemas
few_shot_demos_multiTurnQA = few_shot_demos_multiTurnQA[:10]  # Test용
# print(f'Multi-Turn Few-Shot Demos: {few_shot_demos_multiTurnQA}')  # Test용
print(f'# 4) Multi-Turn : Few-Shot (10) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_multiTurnQA, schemas=schemas, isMultiTurn=True, label=labels[5], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[5]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# PLOT
print('Saving Plot...'); sys.stdout.flush()
compareNplot(precisions, recalls, f1s, all_error_cases=all_error_cases, labels=labels, model=model)
print('Test Complete'); sys.stdout.flush()


########################################################################################################################################################################
print('=========='*20); sys.stdout.flush()
print('=========='*20); sys.stdout.flush()


""" LLaMA 3.1 8B BASE """
# Test Cases
precisions = []
recalls = []
f1s = []
all_error_cases = {}

model = 'LLaMA3.1-8B [BASE]'

# 1) Single-Turn : Zero-Shot + Schemas
print(f'# 1) Single-Turn : Zero-Shot + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=None, schemas=schemas, label=labels[0], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=None)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[0]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 2) Multi-Turn : Zero-Shot + Schemas
print(f'# 2) Multi-Turn : Zero-Shot + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=None, schemas=schemas, isMultiTurn=True, label=labels[1], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=None)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[1]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 3) Single-Turn : One-Shot (1) + Schemas
print(f'# 3) Single-Turn : One-Shot (1) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_singleTurn[:1], schemas=schemas, label=labels[2], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=None)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[2]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 4) Multi-Turn : One-Shot (1) + Schemas
few_shot_demos_multiTurnQA = few_shot_demos_multiTurnQA[:1]  # Test용
# print(f'Multi-Turn Few-Shot Demos: {few_shot_demos_multiTurnQA}')  # Test용
print(f'# 4) Multi-Turn : One-Shot (1) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_multiTurnQA, schemas=schemas, isMultiTurn=True, label=labels[3], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=None)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[3]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 5) Single-Turn : Few-Shot (10) + Schemas'
print(f'# 5) Single-Turn : Few-Shot (10) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_singleTurn[:10], schemas=schemas, label=labels[4], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=None)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[4]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 6) Multi-Turn : Few-Shot (10) + Schemas
few_shot_demos_multiTurnQA = few_shot_demos_multiTurnQA[:10]  # Test용
# print(f'Multi-Turn Few-Shot Demos: {few_shot_demos_multiTurnQA}')  # Test용
print(f'# 4) Multi-Turn : Few-Shot (10) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_multiTurnQA, schemas=schemas, isMultiTurn=True, label=labels[5], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=None)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[5]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# PLOT
print('Saving Plot...'); sys.stdout.flush()
compareNplot(precisions, recalls, f1s, all_error_cases=all_error_cases, labels=labels, model=model)
print('Test Complete'); sys.stdout.flush()


########################################################################################################################################################################
print('=========='*20); sys.stdout.flush()
print('=========='*20); sys.stdout.flush()

""" GPT4o - Mini"""
# Test Cases
precisions = []
recalls = []
f1s = []
all_error_cases = {}

model = 'GPT4o-Mini'

with open('./Test/api_key.txt', 'r') as key_file:
    api_key = key_file.readline().strip()


# 1) Single-Turn : Zero-Shot + Schemas
print(f'# 1) Single-Turn : Zero-Shot + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=None, schemas=schemas, label=labels[0], model=model, openai_api_key=api_key, batch_size=5)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[0]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 2) Multi-Turn : Zero-Shot + Schemas
print(f'# 2) Multi-Turn : Zero-Shot + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=None, schemas=schemas, isMultiTurn=True, label=labels[1], model=model, openai_api_key=api_key, batch_size=5)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[1]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 3) Single-Turn : One-Shot (1) + Schemas
print(f'# 3) Single-Turn : One-Shot (1) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_singleTurn[:1], schemas=schemas, label=labels[2], model=model, openai_api_key=api_key, batch_size=5)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[2]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 4) Multi-Turn : One-Shot (1) + Schemas
few_shot_demos_multiTurnQA = few_shot_demos_multiTurnQA[:1]  # Test용
# print(f'Multi-Turn Few-Shot Demos: {few_shot_demos_multiTurnQA}')  # Test용
print(f'# 4) Multi-Turn : One-Shot (1) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_multiTurnQA, schemas=schemas, isMultiTurn=True, label=labels[3], model=model, openai_api_key=api_key, batch_size=5)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[3]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 5) Single-Turn : Few-Shot (10) + Schemas'
print(f'# 5) Single-Turn : Few-Shot (10) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_singleTurn[:10], schemas=schemas, label=labels[4], model=model, openai_api_key=api_key, batch_size=5)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[4]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 6) Multi-Turn : Few-Shot (10) + Schemas
few_shot_demos_multiTurnQA = few_shot_demos_multiTurnQA[:10]  # Test용
# print(f'Multi-Turn Few-Shot Demos: {few_shot_demos_multiTurnQA}')  # Test용
print(f'# 4) Multi-Turn : Few-Shot (10) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_multiTurnQA, schemas=schemas, isMultiTurn=True, label=labels[5], model=model, openai_api_key=api_key, batch_size=5)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[5]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# PLOT
print('Saving Plot...'); sys.stdout.flush()
compareNplot(precisions, recalls, f1s, all_error_cases=all_error_cases, labels=labels, model=model)
print('Test Complete'); sys.stdout.flush()