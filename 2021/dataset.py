import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import json
import utils

class T5QGDataset(Dataset):
    def __init__(self,data_set_file_path, subtitles_file_path, tokenizer, max_len = 256, ignore_index=-100, data_set_type='train', examples=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_index = self.tokenizer.pad_token_id
        self.mask_index = self.tokenizer.mask_token_id
        self.ignore_index = ignore_index
        if examples:
            self.examples = examples
        else:
            self.examples = self.load_json(data_set_file_path, data_set_type)

        self.subtitles_list = utils.load_subtitles(subtitles_file_path)

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]
            inputs[self.max_len - 1] = self.tokenizer.eos_token_id 

        return inputs

    def truncate_input_ids(self, first, second, prefix_len):
        if len(first) + len(second) > self.max_len:
            first = first[:self.max_len - len(second) - prefix_len]

        return first, second

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs
    
    def load_json(self, file_path, data_type='train'):
        
        with open(file_path, 'r') as f:
            data_set = json.load(f)
            examples = data_set[data_type]

        return examples

    def __getitem__(self, idx):
        example = self.examples[idx]

        question = ""
        if "question" in example.keys():
            question = example['question']
        
        if "answer" in example.keys():
            answer = example['answer']
        elif "False_premise" in example.keys():
            answer = example['False_premise']
        else:
            answer = ""

        vid = example['vid']
        subtitles = utils.get_subtitles_by_vid(self.subtitles_list, vid)
        
        if subtitles: 
            subtitles = " ".join(subtitles)
        else:
            subtitles = ""

        question_input_ids = self.tokenizer.encode(question, add_special_tokens=False)
        answer_input_ids = self.tokenizer.encode(answer, add_special_tokens=False)
        subtitles_input_ids = self.tokenizer.encode(subtitles, add_special_tokens=False)

        prefix_subtitles_token_id = self.tokenizer.encode('subtitles:', add_special_tokens=False)
        prefix_answer_token_id = self.tokenizer.encode('answer:', add_special_tokens=False)
        prefix_question_token_id = self.tokenizer.encode('question:', add_special_tokens=False)

        input_ids = prefix_answer_token_id + answer_input_ids + [self.tokenizer.eos_token_id] + subtitles_input_ids + [self.tokenizer.eos_token_id]
        
        input_ids = self.add_padding_data(input_ids)

        label_ids = prefix_question_token_id
        label_ids = self.tokenizer.encode(question, add_special_tokens=False)
        label_ids.append(self.tokenizer.eos_token_id)
        
        dec_input_ids = [self.tokenizer.bos_token_id]
        dec_input_ids += label_ids[:-1]
        
        dec_input_ids = self.add_padding_data(dec_input_ids)
        label_ids = self.add_ignored_data(label_ids)


        return {'input_ids': np.array(input_ids, dtype=np.int_),
                'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
                'labels': np.array(label_ids, dtype=np.int_)}

    def __len__(self):
        return len(self.examples)
