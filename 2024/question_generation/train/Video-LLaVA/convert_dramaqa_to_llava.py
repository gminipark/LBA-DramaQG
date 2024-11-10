import json
import os
import re
from argparse import ArgumentParser

import dramaqa_utils
from glob import glob

def build_prompt_chatbot(problems, is_test=False):
    examples = {}
    # idx = 0
    for idx, problem in enumerate(problems):
        if 'answer_meta' in problem.keys():
            answer_meta = problem['answer_meta']
            if len(answer_meta) > 0:
                qid= problem['qid']
                try:
                    triple_set = " ".join(list(answer_meta[0].values()))
                except:
                    if len(answer_meta) > 1:
                        triple_set = " ".join(list(answer_meta[1].values()))
                    else:
                        continue
                question = problem['que']
                
                input = triple_set
                output = question
                
                user_prompt = f"Generate a quetion about \"{input}\"."
                assistant_prompt = f"{output}"
                
                examples[idx] = user_prompt, assistant_prompt
            # idx = idx + 1
    return examples


def convert_to_llava(base_dir, file_name):
    
    problems = json.load(open(os.path.join(base_dir, file_name)))
    
    split_problems = build_prompt_chatbot(
        problems, is_test=False)

    target_format = []
    for prob_id, (input, output) in split_problems.items():
        raw_prob_data = problems[prob_id]
        if raw_prob_data['vid'] is None:
            target_format.append({
                "id": prob_id,
                "conversations": [
                    {'from': 'human', 'value': f"{input}"},
                    {'from': 'gpt', 'value': f"{output}"},
                ],
            })

        else:
            episode_id, scene_id, shot_id = dramaqa_utils.get_info_from_vid(raw_prob_data['vid'])
            image_folder = os.path.join(base_dir, "AnotherMissOh_images/dramaqa_frames")
            if shot_id == "0000":
                video_dir = os.path.join(image_folder, f"AnotherMissOh{episode_id}", scene_id)
                image_list = glob(os.path.join(video_dir, '**/*.jpg'), recursive=True)
                image_list = [file_name.replace(image_folder+'/', '') for file_name in image_list]
            else:
                video_dir = os.path.join(image_folder,f"AnotherMissOh{episode_id}", scene_id, shot_id)
                image_list = glob(os.path.join(video_dir, '*.jpg'))
                image_list = [file_name.replace(image_folder+'/', '') for file_name in image_list]
            
            image_len = len(image_list)
            
            if image_len > 16:
                max_image_len = 16
            else:
                max_image_len = image_len
    
            target_format.append({
                "id": prob_id,
                "image": image_list,
                "conversations": [
                    {'from': 'human', 'value': '<image>' * max_image_len + f"\n{input}"},
                    {'from': 'gpt', 'value': f"{output}"},
                ],
            })

    print(f'Number of samples: {len(target_format)}')
    print(target_format[0])

    if "train" in file_name:
        type = 'train'
    elif "val" in file_name:
        type = 'val'
    else:
        type = 'test'
    
    with open(os.path.join(base_dir, f"llava_dramaqg_{type}.json"), "w") as f:
        json.dump(target_format, f, indent=2)


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--file_name', type=str, required=True)
    
    args = parser.parse_args()

    convert_to_llava(args.base_dir, args.file_name)