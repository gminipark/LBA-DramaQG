from openai import OpenAI
import os
import re
import json
from typing import List, Tuple
import time
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv
from tqdm import tqdm
from utils_for_classification import load_descriptions, make_relation_list_lines, build_prompt_for_relation, build_prompt_for_description, normalize_relation_name

from utils_for_novel_relation import (
    load_descriptions,
    make_relation_list_lines,
    build_prompt_for_relation,
    build_prompt_for_description,
    normalize_relation_name,
)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
gpt_client = OpenAI(api_key=api_key)  


INITIAL_DESC_CSV = "./relation_35_description_w_no_relation.csv"
SAMPLES_CSV = "./gpt_known_novel_unclear_with_description_ver2.csv"
OUTPUT_DIR = "./output"
MODEL_NAME = "gpt-5"
SAVE_EVERY = 10  

os.makedirs(OUTPUT_DIR, exist_ok=True)
PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "no_rel_in_descriptions_result.csv")
UPDATED_DESC_CSV = os.path.join(OUTPUT_DIR, "updated_no_rel_in_descriptions_description.csv")


class OpenAIWrapper:
    def __init__(self, model: str):
        self.model = model  

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type(Exception),
    )
    def complete(self, prompt: str) -> str:
        resp = gpt_client.responses.create(
            model=self.model,
            input=prompt,
            reasoning={"effort": "high"},
            text={"verbosity": "low"}
        )
        try:
            return resp.output_text.strip()
        except Exception:
            return json.dumps(resp.model_dump())


def process_samples(
    samples_df: pd.DataFrame,
    desc_df: pd.DataFrame,
    client: OpenAIWrapper,
    save_every: int = SAVE_EVERY,
    predictions_csv: str = PREDICTIONS_CSV,
    updated_desc_csv: str = UPDATED_DESC_CSV,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    records = []
    desc_df_current = desc_df.copy().reset_index(drop=True)

    total = len(samples_df)
    known_cnt = 0
    novel_cnt = 0
    skipped_cnt = 0
    t0 = time.time()

    for i, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc="Processing"):
        sentence = str(row["text"]).strip()
        subject = str(row["subject"]).strip()
        obj = str(row["object"]).strip()

        rel_lines = make_relation_list_lines(desc_df_current)
        prompt = build_prompt_for_relation(sentence, subject, obj, rel_lines)  

        retry_attempts = 3
        raw = ""
        for attempt in range(1, retry_attempts + 1):
            try:
                raw = client.complete(prompt) 
            except Exception as e:
                raw = f"ERROR: {e}"

            if raw and str(raw).strip() != "":
                break

            print(f"GPT 응답이 비어 있음 index={i}, 시도={attempt}/{retry_attempts}, 입력 문장='{sentence}'")
            if attempt < retry_attempts:
                time.sleep(5)

        if not raw or str(raw).strip() == "":
            print(f"GPT 응답이 최종적으로 비어 있음. index={i}, 입력 문장='{sentence}'")
            skipped_cnt += 1
            continue
        elif str(raw).startswith("ERROR:"):
            print(f"[에러] GPT 호출 오류 발생: {raw}")


        pred_relation = normalize_relation_name(raw) if not raw.startswith("ERROR:") else ""
        if not pred_relation:
            pred_relation = f"novel:auto_{i}"

        existing_relations = set(desc_df_current["relation"].tolist()) 
        is_novel = pred_relation not in existing_relations 

        if is_novel:
            novel_cnt += 1
            try:
                print('새로운 relation 추가 :', pred_relation)
                d_prompt = build_prompt_for_description(pred_relation, desc_df_current) 
                desc_text = client.complete(d_prompt).strip() 
                print('새로 생성된 description :',desc_text)
                print('=============================================================')
            except Exception as e:
                desc_text = f"Description generation failed: {e}"

            desc_df_current = pd.concat(
                [desc_df_current, pd.DataFrame([{"relation": pred_relation, "description": desc_text}])],
                ignore_index=True,
            )
        else:
            known_cnt +=1

        rec = row.to_dict()  
        rec.update({
            "predicted_relation": pred_relation,
            "status": "novel" if is_novel else "known",
            "continual_description": raw  
        })
        records.append(rec)

        if (i + 1) % save_every == 0:
            pd.DataFrame.from_records(records).to_csv(predictions_csv, index=False)
            desc_df_current.to_csv(updated_desc_csv, index=False)

            elapsed = time.time() - t0
            pct = (i + 1) / total * 100 if total else 0
            print(f"[진행] {i+1}/{total} ({pct:.1f}%) | known={known_cnt}, novel={novel_cnt}, skipped={skipped_cnt} | {elapsed:.1f}s 경과")

    preds_df = pd.DataFrame.from_records(records)
    preds_df.to_csv(predictions_csv, index=False)
    desc_df_current.to_csv(updated_desc_csv, index=False)
    return preds_df, desc_df_current


def main():
    desc_df = load_descriptions(INITIAL_DESC_CSV)
    samples_df = pd.read_csv(SAMPLES_CSV)
    for col in ["text", "subject", "object"]:
        if col not in samples_df.columns:
            raise ValueError(f"SAMPLES_CSV must include column '{col}'")

    samples_df = samples_df.sample(frac=1, random_state=42).reset_index(drop=True) 

    client = OpenAIWrapper(model=MODEL_NAME)

    preds_df, updated_desc_df = process_samples(
        samples_df=samples_df,
        desc_df=desc_df,
        client=client,
        save_every=SAVE_EVERY,
        predictions_csv=PREDICTIONS_CSV,
        updated_desc_csv=UPDATED_DESC_CSV,
    )

    print(f"Saved predictions -> {PREDICTIONS_CSV}")
    print(f"Saved updated descriptions -> {UPDATED_DESC_CSV}")


if __name__ == "__main__":
    main()
