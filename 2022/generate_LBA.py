import torch
import T5
import dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data file path
data_set_file_path = 'DramaQG/fine-tuning.json'
subtitles_file_path = "DramaQA/AnotherMissOh_script.json"
model_path = "output_fine_t5_base_lr_3e-5_qg/t5_best.pth"
output_path = 'outputs_LBA'

if not os.path.exists(output_path):
    os.mkdir(output_path)

# config
batch_size = 64

# model. tokenizer init
model = T5.T5ConditionalGeneration().to(device)
tokenizer = model.tokenizer

input_examples = [{"vid" : "AnotherMissOh14_001_0000",
                "False_promise": "Haeyoung1 and Dokyung are in love and the two went through many things before starting to date.",
                }]

# dataset
dev_dataset = dataset.T5QGDataset(data_set_file_path=data_set_file_path, 
                                    subtitles_file_path=subtitles_file_path, 
                                    tokenizer=tokenizer, 
                                    data_set_type='val',
                                    examples=input_examples)
dev_dataloader = DataLoader(dev_dataset, batch_size)


model.load_state_dict(torch.load(model_path))    
model.eval()

with open(os.path.join(output_path, f'outputs.json'), 'w', encoding='utf-8') as f:
    outputs = []
    for step_index, batch_data in tqdm( enumerate(dev_dataloader), f"[GENERATE]", total=len(dev_dataloader)):

        input_ids, decoder_input_ids, labels = tuple(value.to(device) for value in batch_data.values())

        output = model.model.generate(input_ids=input_ids, eos_token_id=tokenizer.eos_token_id, max_length=100, num_beams=5)

        for o in output:
            o = tokenizer.decode(o, skip_special_tokens=True)
            o = o.replace(' ##', '').replace('##', '').strip()
            outputs.append({'question' : o})


    json.dump(outputs, f)
