import os
import re
import json
from typing import List, Tuple
import time
import pandas as pd
from tqdm import tqdm

def load_descriptions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {"relation", "description"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Description CSV must have columns {required_cols}, found: {list(df.columns)}")
    df["relation"] = df["relation"].astype(str).str.strip()
    df["description"] = df["description"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["relation"], keep="first").reset_index(drop=True)
    return df


def make_relation_list_lines(desc_df: pd.DataFrame) -> List[str]:
    return [f"relation: {row['relation']}, description: {row['description']}" for _, row in desc_df.iterrows()]


def build_prompt_for_relation(sentence: str, subject: str, obj: str, rel_lines: List[str]) -> str:
    rel_block = "\n".join(rel_lines)
    prompt = (
        "You are an expert annotator for relation extraction tasks.\n"
        "You need to determine whether a sentence expresses a predefined relation or a new relation.\n"
        "Below are the predefined relation names and their corresponding descriptions.\n\n"
        f"{rel_block}\n\n"
        "Classify, based on the relation list and their descriptions above, whether the relation between the subject\n"
        "If the relation exists in the list, specify which predefined relation it corresponds to.\n"
        "If the relation does not exist in the list, define a new relation name by following the format of the given relation list.\n"
        f"Sentence: {sentence}\n"
        f"Subject: {subject}\n"
        f"Object: {obj}\n"
        f"Output format: <relation name>\n"
        "relation name : "
    )
    return prompt


def build_prompt_for_description(novel_relation_name: str, desc_df: pd.DataFrame) -> str:
    example_lines = []
    for _, row in desc_df.iterrows():
        example_lines.append(
            f"description of {row['relation']}: {row['description']}"
        )
    examples_block = "\n".join(example_lines)

    prompt = (
        "You are an expert annotator for relation extraction tasks.\n"
        "Below are the descriptions of predefined relations.\n"
        f"{examples_block}\n\n"
        "Generate a description for the new relation in 5 to 10 sentences, referring to the style and format of the above descriptions as guidance.\n"
        "The description of the novel relation must include characteristics that clearly distinguish it from existing relations.\n"
        "The description you create will be used for a relation extraction task involving two entities within a sentence.\n\n"
        f"novel relation name: {novel_relation_name}\n"
        f"Your description of {novel_relation_name}:"
    )
    return prompt


def normalize_relation_name(raw: str) -> str:
    if raw is None:
        return ""
    line = str(raw).splitlines()[0].strip()
    line = re.sub(r"^(relation\s*name|name|relation)\s*[:\-]\s*", "", line, flags=re.IGNORECASE).strip()
    line = re.sub(r"^[\"'`<\(\[\{]+", "", line)
    line = re.sub(r"[\"'`>\)\]\}]+$", "", line)
    line = re.sub(r"[,\.\s]+$", "", line)
    return line
