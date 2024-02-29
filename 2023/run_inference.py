import os
import argparse
from tqdm import tqdm
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from dataset import ImageTextGenerationDataset
from utils import convert_inputs_to_examples


device = "cuda" if torch.cuda.is_available() else "cpu"

def move_to(batch, device):

    for batch_key, batch_value in list(batch.items()):
        if torch.is_tensor(batch_value):
            if batch_key == "input_ids":
                batch[batch_key] = batch_value.to(device)
            else:
                batch[batch_key] = batch_value.to(device, torch.float16)
    return batch

def get_model(args):
    
    cache_dir = args.cache_dir if args.cache_dir else os.path.join("./", args.model_name_or_path.split("/")[-1])
    
    model = Blip2ForConditionalGeneration.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, cache_dir=cache_dir)
    
    return model

def get_processor(args):
    
    cache_dir = args.cache_dir if args.cache_dir else os.path.join("./", args.model_name_or_path.split("/")[-1])
    processor_name_or_path = args.processer_name_or_path if args.processer_name_or_path else args.model_name_or_path
    
    processor = Blip2Processor.from_pretrained(processor_name_or_path, cache_dir=cache_dir)
    
    return processor

def get_decoding_strategy(strategy_name):
    
    
    if strategy_name == "greedy":
        parameters = {"max_new_tokens" : 20}
        
    elif strategy_name == "beam":
        parameters = {"max_new_tokens" : 20,
                      "num_beams" : 4}
        
    elif strategy_name == "constrastive":
        parameters = {"max_new_tokens" : 20,
                      "penalty_alpha" : 0.6,
                      "top_k":4}
        
    elif strategy_name == "diverse":
        parameters = {"max_new_tokens" : 20,
                      "num_beams" : 4, 
                      "num_beam_groups" : 4,
                      "diversity_penalty" : 1.0}
        
    elif strategy_name == "sample":
        parameters = {"max_new_tokens" : 20,
                      "do_sample" : True}
    return parameters
def add_args(parser):
    
    parser.add_argument("--model_name_or_path", type=str, default="Salesforce/blip2-flan-t5-xxl")
    parser.add_argument("--processer_name_or_path", type=str)
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--prompt_type", default='0', choices=['0', '1', '2'])
    parser.add_argument("--decoding_strategy",default='beam', choices=['greedy', 'beam', 'constrastive', 'diverse', 'sample'])
    parser.add_argument("--input_path", type=str, default="../../Integration-Outputs/output_KAIST.json", required=True)
    parser.add_argument("--output_path", type=str, default="../../Integration-Outputs/output_KHU.json")
    
    return parser


def main():

    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    
    args = parser.parse_args()
    
    model = get_model(args)
    processor = get_processor(args)
    
    # unanswerable question and uncertain_information(object)
    with open(args.input_path, 'r') as f:
        loaded_json = json.load(f)
    
    examples = convert_inputs_to_examples(loaded_json)
    
    test_dataset = ImageTextGenerationDataset(args, examples, processor)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn=test_dataset.collate_fn)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.to(device)
    model.eval()

    generated_questions = []
    results = []
    example_qid = examples[0]['qid']

    for i, batch in enumerate(tqdm(test_dataloader)):
        if example_qid != examples[i]['qid']:
            output = {  
                    'qid': examples[i-1]['qid'],
                    'vid': examples[i-1]['vid'],
                    'main_question': examples[i-1]['question'],
                    'sub_questions': generated_questions,
                }
            results.append(output)
            generated_questions = []
        example_qid = examples[i]['qid']
        
        inputs = move_to(batch, device)
        
        outputs = []
        
        parameters = get_decoding_strategy(args.decoding_strategy)
        
        outputs += model.generate(**inputs, **parameters)
        
        additional_questions = processor.batch_decode(outputs, skip_special_tokens=True)
        
        generated_questions += additional_questions
        
    json.dump(results, open(args.output_path, 'w'), indent=4)
    
if __name__ == "__main__":
    
    main()


