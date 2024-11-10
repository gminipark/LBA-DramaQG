import math
import os
import argparse
import json
import numpy as np
from PIL import Image

import torch
import transformers
from tqdm import tqdm
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN
from videollava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from videollava.model.builder import load_pretrained_model
from videollava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from videollava.train.train import smart_tokenizer_and_embedding_resize
from evaluate import load

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--cache_dir', help='', required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)

    return parser.parse_args()

def get_model_output(model, video_processor, tokenizer, video, qs, args):

    # qs = DEFAULT_IMAGE_TOKEN*16 +'./'+ qs.replace(DEFAULT_IMAGE_TOKEN,'')

    conv_mode = "llava_v1"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

   
    video_tensor = video_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().to(args.device)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    b, c, w, h = video_tensor.shape 
    video_tensor = video_tensor.reshape(c, b, w, h) # (b, c, w, h)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[video_tensor],
            # do_sample=True,
            max_new_tokens=512,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    
    return outputs


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    model = model.to(args.device)

    # Load both ground truth file containing questions and answers
    # with open(args.gt_file_question) as file:
    #     gt_questions = json.load(file)
    # with open(args.gt_file_answers) as file:
    #     gt_answers = json.load(file)

    gt_dataset = json.load(open(args.gt_file, "r"))
    # gt_answers = get_chunk(gt_answers, args.num_chunks, args.chunk_idx)
    gt_dataset =gt_dataset
    
    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(answers_file, "w")

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    pred_list = []  # List to store the output results
    label_list = []

    # video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    index = 0
    for sample in tqdm(gt_dataset):
        video_path = sample['image']
        
        input = sample['conversations'][0]['value']
        label = sample['conversations'][1]['value']
        
        label_list.append(label)

        sample_set = {'input': input, 'output': label}

        # Load the video file
        # Added this line
        total_frames = len(video_path)
        indices = np.arange(0, total_frames, total_frames / 16).astype(int).tolist()
        images = []
        for idx in indices:
            images.append(os.path.join(args.video_dir, video_path[idx]))
        
        # try:
        # Run inference on the video and add the output to the list
        pred = get_model_output(model, processor['image'], tokenizer, images, input, args)
        sample_set['pred'] = pred
        pred_list.append(pred)
        # except Exception as e:
        #     print(f"Error processing video file '{video_name}': {e}")
        
        ans_file.write(json.dumps(sample_set) + "\n")

    ans_file.close()
    
    
    bleu = load('bleu')
        
    scores = bleu.compute(predictions=pred_list, references=label_list)
    
    with open(os.path.join(args.output_dir, "metrics.json"), 'w') as f:
        json.dump(scores, f)    
    
    # Save the output list to a JSON file
    # with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
    #     json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
