import torch
from torch.utils.data import  Dataset

from PIL import Image

from image_utils import get_image_from_vid


prompts = {"0" : "Instructions: Given a picture, a question and a ambiguous entity of the question are provided.\
        An answer can not be inferred by the ambiguous entity. \
        The ambiguous entity(object) of question  which is existed in the picture is important key to infer the answer. \
        The additional information of the ambiguous entity  is helpful for answer the question more correctly. \
        Therefore, our goal is to get new information about the ambiguous entity related to answer by asking a new question. \
        Generate an additional question about ambiguous entity to help to answer the original question correctly. " ,
        "1" : "Instructions: Given a picture, a question and a ambiguous entity of the question are provided.\
        An answer can not be inferred by the ambiguous entity. \
        The ambiguous entity(object) of question  which is existed in the picture is important key to infer the answer. \
        The additional information of the ambiguous entity  is helpful for answer the question more correctly. \
        Therefore, our goal is to get new information about the ambiguous entity related to answer by asking a new question. \
        Generate an additional question about the location of the ambiguous entity to help to answer the original question correctly. ",
        "2" : "Instructions: Given a picture, a question and a ambiguous entity of the question are provided.\
        An answer can not be inferred by the ambiguous entity. \
        The ambiguous entity(object) of question  which is existed in the picture is important key to infer the answer. \
        The additional information of the ambiguous entity  is helpful for answer the question more correctly. \
        Therefore, our goal is to get new information about the ambiguous entity related to answer by asking a new question. \
        Generate an additional question about an attribute of the ambiguous entity to help to answer the original question correctly. "}

class ImageTextGenerationDataset(Dataset):
    def __init__(self, args, examples, processor):
        self.args = args
        self.examples = examples
        self.processor = processor

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        
        images = get_image_from_vid(self.args.image_dir, item["vid"])
        
        input_image = Image.open(images[0])
        
        encoding = self.processor(images=input_image, padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        
        question = prompts[self.args.prompt_type] + \
        "Original question: " + item['question'] + " " + \
        "Ambiguous entity: " + item['ambiguous_object'] + " " + \
        "Additional question: "
        encoding["text"] = question
        
        return encoding


    def collate_fn(self, batch):
        # pad the input_ids and attention_mask
        processed_batch = {}
        for key in batch[0].keys():
            if key == "text":
                text_inputs = self.processor.tokenizer(
                    [example["text"] for example in batch], padding=True, return_tensors="pt"
                )
                processed_batch["input_ids"] = text_inputs["input_ids"]
                processed_batch["attention_mask"] = text_inputs["attention_mask"]
            else:
                processed_batch[key] = torch.stack([example[key] for example in batch])
        
        
        return processed_batch