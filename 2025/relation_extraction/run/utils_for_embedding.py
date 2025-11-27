import torch
import re

def extract_all_tags(text_series):
    tag_pattern = re.compile(r"<\\?e[12]:[A-Za-z_]+>")
    all_tags = set()
    for t in text_series.astype(str):
        all_tags.update(tag_pattern.findall(t))
    return all_tags


def embed_batch(text_list, tokenizer, MAX_LENGTH, model, open_tag_ids, device):
    with torch.no_grad():
        enc = tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state 

        embeddings = []
        input_ids_cpu = input_ids.detach().cpu().numpy()

        for b in range(input_ids.shape[0]):
            ids = input_ids_cpu[b]

            present_open_ids = [tid for tid in open_tag_ids.values() if tid in ids]

            e1_pos = None
            e2_pos = None

            for pos, tok_id in enumerate(ids):
                tok = tokenizer.convert_ids_to_tokens(int(tok_id))
                if e1_pos is None and tok.startswith("<e1:") and tok.endswith(">"):
                    e1_pos = pos
                elif e2_pos is None and tok.startswith("<e2:") and tok.endswith(">"):
                    e2_pos = pos
                if e1_pos is not None and e2_pos is not None:
                    break

            if e1_pos is None:
                e1_vec = torch.zeros(last_hidden.size(-1), device=device)
            else:
                e1_vec = last_hidden[b, e1_pos, :]

            if e2_pos is None:
                e2_vec = torch.zeros(last_hidden.size(-1), device=device)
            else:
                e2_vec = last_hidden[b, e2_pos, :]

            emb = torch.cat([e1_vec, e2_vec], dim=-1) 
            embeddings.append(emb.detach().cpu().numpy())

        return embeddings