import random
import pandas as pd
import os, json
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from utils_for_descriptions import gather_examples, build_prompt_for_relation

random.seed(42)

# 5-shot sampling
D_u_410 = pd.read_csv("./sampled_oneshot_example_gpt.csv")
D_u = pd.read_csv("./sampled_with_gpt.csv")

ids_in_410 = set(D_u_410['id'])

known_relations = [
    "per:city_of_birth",
    "org:stateorprovince_of_headquarters",
    "org:member_of",
    "per:date_of_death",
    "per:city_of_death",
    "per:children",
    "no_relation"
]


D_u_filtered = D_u[
    (~D_u['id'].isin(ids_in_410))
]

target_relations = known_relations

D_u_target = D_u_filtered[
    D_u_filtered['relation'].isin(target_relations)
]

D_u_examples = (
    D_u_target
    .groupby('relation', group_keys=False)
    .apply(lambda x: x.sample(n=5, random_state=62) if len(x) >= 5 else x)
    .reset_index(drop=True)
)

print(D_u_examples.shape)
print(D_u_examples['relation'].value_counts())

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


save_path = "./relation_35_descriptions_ver2.csv"
k_examples = 5
random_state = 42


relations = sorted(D_u_examples["relation"].astype(str).unique())

results = []

for rel in tqdm(relations, desc="Describing relations"):
    examples = gather_examples(D_u_examples, rel, k=k_examples, seed=random_state)
    prompt = build_prompt_for_relation(rel, examples)

    result = client.responses.create(
        model="gpt-5",
        input=prompt,
        reasoning={"effort": "high"},
        text={"verbosity": "high"},
    )

    answer = getattr(result, "output_text", None)
    if answer is None and hasattr(result, "output"):
        answer = str(result.output[0].content[0].text)
    if answer is None:
        answer = ""

    description = answer.strip().strip('"').splitlines()[0] if answer else ""
    results.append({"relation": rel, "description": description})

    temp_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    temp_df.to_csv(save_path, index=False)
    print(f" Saved progress ({len(results)}/{len(relations)}) to {save_path}\n")

out_df = pd.DataFrame(results)
out_df.to_csv(save_path, index=False)
print(f"\nAll {len(out_df)} relation descriptions saved to {save_path}")
