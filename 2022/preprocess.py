import argparse
import json 
import random
import os
from tqdm import tqdm 
from utils import set_random_seed

def load_qa(path):
    
    examples = []
    with open(path, 'r') as f:
        
        qas = json.load(f)

        for qa in tqdm(qas):
            example = {}
            
            example["vid"] = qa['vid']
            example["answer"] = qa['answers'][qa['correct_idx']]
            example["question"] = qa["que"]
            
            examples.append(example)

    return examples

def split_data_set(original_set, ratio):

    random.shuffle(original_set)

    split_point = int(len(original_set) * ratio)
    part01 = original_set[:split_point]
    part02 = original_set[split_point:]
    
    return (part01, part02)

def save_dataset(data_set, args):

    project_dir = os.path.dirname(args.directory_path)
    
    data_set_dir = os.path.join(project_dir,"DramaQG")
    
    if "DramaQG" not in os.listdir(project_dir):
        os.mkdir(data_set_dir)
    
    
    for data_set_type, data_set_value in data_set.items():
        
        file_path = os.path.join(data_set_dir, data_set_type + ".json")
        with open(file_path, 'w') as f:
            json.dump(data_set_value, f)


def preprocess():
    parser = argparse.ArgumentParser()

    parser.add_argument("--directory_path", required=True, help="The directory path of train set")

    parser.add_argument("--split_train_ratio", default=0.9, help="The ratio to split original train set for post-training")

    parser.add_argument("--split_valid_ratio", default=0.5, help="The ratio to split original valid set for fine-tuning")

    parser.add_argument("--seed", default=1, help="The seed for fixing dataset")

    args = parser.parse_args()

    set_random_seed(1)

    origin_train_examples = load_qa(os.path.join(args.directory_path, "AnotherMissOhQA_train_set.json"))
    origin_val_examples = load_qa(os.path.join(args.directory_path, "AnotherMissOhQA_val_set.json"))
    

    post_train_examples, post_val_examples = split_data_set(origin_train_examples, args.split_train_ratio)

    train_examples = origin_train_examples
    val_examples, test_examples = split_data_set(origin_val_examples, args.split_valid_ratio)

    data_set = {}

    data_set['post-training'] = {}
    data_set['post-training']['train'] = post_train_examples
    data_set['post-training']['val'] = post_val_examples
    data_set['fine-tuning'] = {}
    data_set['fine-tuning']['train'] = train_examples
    data_set['fine-tuning']['val'] = val_examples
    data_set['fine-tuning']['test'] = test_examples

    save_dataset(data_set,args)


if __name__ == "__main__":
    preprocess()