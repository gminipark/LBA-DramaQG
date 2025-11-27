import pandas as pd
import json
from pathlib import Path
from utils_for_tagging import get_tags, tag_from_json
classification_result = pd.read_csv('./output/no_rel_in_descriptions_result.csv')

known_relation_list = [
    "per:city_of_birth",
    "org:stateorprovince_of_headquarters",
    "org:member_of",
    "per:date_of_death",
    "per:city_of_death",
    "per:children",
    "no_relation"
]

classification_result['classification_label'] = classification_result['predicted_relation'].apply(
    lambda x: 'known' if x in known_relation_list else 'novel'
)

merged_df = classification_result

paths = [
    "/workspace/KG-LLM/Data/TACRED/data/json/train.json",
    "/workspace/KG-LLM/Data/TACRED/data/json/test.json",
    "/workspace/KG-LLM/Data/TACRED/data/json/dev.json",
]

all_samples = []
for p in paths:
    with open(p, "r", encoding="utf-8") as f:
        all_samples.extend(json.load(f))

id2sample = {ex["id"]: ex for ex in all_samples}

print(f"TACRED 데이터 로드 완료: 총 {len(id2sample)}개 샘플")


TAG_STYLE = "backslash"  

if "id" not in merged_df.columns:
    raise ValueError("merged_df에 'id' 컬럼 필요")

merged_df["tagged_text"] = merged_df["id"].apply(lambda x: tag_from_json({"id": x}, id2sample, TAG_STYLE))

print("=== 태깅 결과 샘플 ===")
print(merged_df[["id", "text", "tagged_text"]].head(10))
print(f"태깅 완료. 태그 생성 실패 행 수: {merged_df['tagged_text'].isna().sum()}")

merged_df.to_csv('./output/tagging_result.csv', index=False)

