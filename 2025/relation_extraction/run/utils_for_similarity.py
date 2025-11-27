import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 128

_device = "cuda" if torch.cuda.is_available() else "cpu"
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModel.from_pretrained(MODEL_NAME).to(_device).eval()


@torch.no_grad()
def embed_texts(texts):
    inputs = _tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(_device)

    outputs = _model(**inputs)
    hidden = outputs.last_hidden_state 
    attn_mask = inputs["attention_mask"].unsqueeze(-1) 

    summed = (hidden * attn_mask).sum(dim=1)
    counts = attn_mask.sum(dim=1).clamp(min=1)
    emb = summed / counts 

    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()
