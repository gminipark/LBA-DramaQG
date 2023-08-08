import torch
import T5
import dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data file path
data_set_file_path = 'DramaQG/fine-tuning.json'
subtitles_file_path = "DramaQA/AnotherMissOh_script.json"
output_path = 'output_fine_t5_base_3e-5_qg'

if not os.path.exists(output_path):
    os.mkdir(output_path)

# config
batch_size = 64

# model. tokenizer init
model = T5.T5ConditionalGeneration().to(device)
tokenizer = model.tokenizer

# dataset
dev_dataset = dataset.T5QGDataset(data_set_file_path=data_set_file_path, 
                                    subtitles_file_path=subtitles_file_path, 
                                    tokenizer=tokenizer, 
                                    data_set_type='val')
dev_dataloader = DataLoader(dev_dataset, batch_size)


count = -1
while(True):
    count += 1
    model.load_state_dict(torch.load(os.path.join(output_path, f't5_epoch_{count}.pth')))
    model.eval()

    with open(os.path.join(output_path, f'output_{count}.txt'), 'w', encoding='utf-8') as f:
        for step_index, batch_data in tqdm( enumerate(dev_dataloader), f"[GENERATE]", total=len(dev_dataloader)):

            input_ids, decoder_input_ids, labels = tuple(value.to(device) for value in batch_data.values())

            output = model.model.generate(input_ids=input_ids, eos_token_id=tokenizer.eos_token_id, max_length=100, num_beams=5)

            for o in output:
                o = tokenizer.decode(o, skip_special_tokens=True)
                o = o.replace(' ##', '').replace('##', '').strip()
                f.write(o+'\n')
