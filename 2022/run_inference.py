import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from dataset import ImageTextGenerationDataset


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

def add_args(parser):
    
    parser.add_argument("--model_name_or_path", type=str, default="Salesforce/blip2-flan-t5-xxl")
    parser.add_argument("--processer_name_or_path", type=str)
    parser.add_argument("--cache_dir", type=str)
    
    parser.add_argument("--image_dir", type=str, required=True)
    
    return parser


def main():

    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    
    args = parser.parse_args()
    
    model = get_model(args)
    processor = get_processor(args)
    
    # unanswerable question and uncertain_information(object)
    samples = [
        {"question" : "Why did Dokyung go to the old man?",
         "uncertain_information" : "man",
         "vid" : "AnotherMissOh17_001_0000"
        },
    ]
    
    
    test_dataset = ImageTextGenerationDataset(args, samples, processor)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn=test_dataset.collate_fn)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.to(device)
    model.eval()

    generated_results = []
    
    for batch in tqdm(test_dataloader):
        
        inputs = move_to(batch, device)
        
        outputs = model.generate(**inputs, max_new_tokens=20)
        
        additional_questions = processor.batch_decode(outputs, skip_special_tokens=True)
        
        generated_results += additional_questions
        
    print(generated_results)
    
if __name__ == "__main__":
    
    main()


