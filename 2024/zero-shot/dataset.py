import torch
from torch.utils.data import Dataset

from PIL import Image
import numpy as np

from image_utils import get_images_from_vid


prompts = {
    "0": "Instructions: Given a picture, a question and a ambiguity of the question are provided.\
        An answer can not be inferred by the ambiguity. \
        The ambiguity of question  which is existed in the picture is important key to infer the answer. \
        The additional information of the ambiguity  is helpful for answer the question more correctly. \
        Therefore, our goal is to get new information about the ambiguity related to answer by asking a new question. \
        Generate an additional question about ambiguity to help to answer the original question correctly. ",
    "1": "Instructions: Given a picture, a question and a ambiguity of the question are provided.\
        An answer can not be inferred by the ambiguity. \
        The ambiguity of question  which is existed in the picture is important key to infer the answer. \
        The additional information of the ambiguity  is helpful for answer the question more correctly. \
        Therefore, our goal is to get new information about the ambiguity related to answer by asking a new question. \
        Generate an additional question about the location of the ambiguity to help to answer the original question correctly. ",
    "2": "Instructions: Given a picture, a question and a ambiguity of the question are provided.\
        An answer can not be inferred by the ambiguity. \
        The ambiguity of question  which is existed in the picture is important key to infer the answer. \
        The additional information of the ambiguity  is helpful for answer the question more correctly. \
        Therefore, our goal is to get new information about the ambiguity related to answer by asking a new question. \
        Generate an additional question about an attribute of the ambiguity to help to answer the original question correctly. ",
}


class VideoTextGenerationDataset(Dataset):
    def __init__(self, args, examples, processor):
        self.args = args
        self.examples = examples
        self.processor = processor

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]

        images = get_images_from_vid(self.args.video_dir, item["vid"])

        video = [Image.open(image) for image in images]

        if len(video) > 8:
            indices = np.arange(0, len(video), len(video) / 8).astype(int)
            video = self.read_video(video, indices)

        video = np.array(video)

        encoding = self.processor.video_processor(images=video, return_tensors="pt")
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        question = (
            prompts[self.args.prompt_type]
            + "Original question: "
            + item["question"]
            + " "
            + "Ambiguity: "
            + item["ambiguity"]
            + " "
            + "Additional question: "
        )

        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{question}"},
                    {"type": "video"},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        encoding["text"] = prompt
        return encoding

    def collate_fn(self, batch):
        # pad the input_ids and attention_mask
        processed_batch = {}
        for key in batch[0].keys():
            if key == "text":
                text_inputs = self.processor.tokenizer(
                    [example["text"] for example in batch],
                    padding=True,
                    return_tensors="pt",
                )
                processed_batch["input_ids"] = text_inputs["input_ids"]
                processed_batch["attention_mask"] = text_inputs["attention_mask"]
            else:
                processed_batch[key] = torch.stack([example[key] for example in batch])

        return processed_batch

    def read_video(self, video, indices):

        images = []
        start_index = indices[0]
        end_index = indices[-1]

        for i, image in enumerate(video):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                images.append(image)

        return images
