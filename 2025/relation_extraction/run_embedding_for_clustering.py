import re
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from utils_for_embedding import extract_all_tags, embed_batch

classification_result = pd.read_csv('./output/tagging_result.csv')

MODEL_NAME = "microsoft/deberta-v3-base"  
MAX_LENGTH = 256                        
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_tags = extract_all_tags(classification_result["tagged_text"])

open_tag_pattern = re.compile(r"<e[12]:[A-Za-z_]+>")
open_tags = sorted({tag for tag in all_tags if open_tag_pattern.fullmatch(tag)})

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModel.from_pretrained(MODEL_NAME)

additional_special_tokens = [t for t in sorted(all_tags) if t not in tokenizer.get_vocab()]
if additional_special_tokens:
    tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
    model.resize_token_embeddings(len(tokenizer))

model.to(device)
model.eval()

open_tag_ids = {t: tokenizer.convert_tokens_to_ids(t) for t in open_tags}

all_embeddings = []
texts = classification_result["tagged_text"].astype(str).tolist()

for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
    batch_texts = texts[i:i + BATCH_SIZE]
    batch_embs = embed_batch(batch_texts, tokenizer, MAX_LENGTH, model, open_tag_ids, device)
    all_embeddings.extend(batch_embs)

assert len(all_embeddings) == len(classification_result)

classification_result = classification_result.copy()
classification_result["embedding"] = all_embeddings


print(classification_result[["tagged_text", "embedding"]].head())
print("임베딩 차원 예시:", classification_result["embedding"].iloc[0].shape)

classification_result.to_csv('./output/embedding_result.csv', index=False)

