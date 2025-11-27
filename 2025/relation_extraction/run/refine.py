import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import os
from tqdm import tqdm
from utils_for_refine import build_prompt, call_gpt

low_similarity_df = pd.read_csv("./output/low_similarity_df.csv")

relation_list = [ # known + novel
    "org:alternate_names","org:city_of_headquarters","org:country_of_headquarters",
    "org:dissolved","org:founded","org:founded_by","org:members",
    "org:number_of_employees/members","org:parents","org:political/religious_affiliation",
    "org:shareholders","org:subsidiaries","org:top_members/employees","org:website",
    "per:age","per:alternate_names","per:cause_of_death","per:charges",
    "per:cities_of_residence","per:countries_of_residence","per:country_of_birth",
    "per:country_of_death","per:date_of_birth","per:employee_of","per:origin",
    "per:other_family","per:parents","per:religion","per:schools_attended",
    "per:siblings","per:spouse","per:stateorprovince_of_birth","per:stateorprovince_of_death",
    "per:stateorprovinces_of_residence","per:title",
    "per:city_of_death" ,"per:children" ,"org:stateorprovince_of_headquarters", "org:member_of","per:city_of_birth","per:date_of_death" 
]
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) 

save_path = "./output/cluster_refine_result.csv"
os.makedirs(os.path.dirname(save_path), exist_ok=True)


outputs = []
indices = []
batch_size = 10  

for idx, row in tqdm(low_similarity_df.iterrows(), total=len(low_similarity_df), desc="Querying ChatGPT"):
    prompt = build_prompt(
        relation_list,
        sentence=row["text"],
        subject=row["subject"],
        obj=row["object"],
        cluster_name = row['cluster_name']
    )
    
    result_text = call_gpt(prompt, client)
    outputs.append(result_text)

    indices.append(idx)

    if (idx + 1) % batch_size == 0:
        for i, pred in zip(indices, outputs):
            low_similarity_df.at[i, "Cluster_refine"] = pred
        
        low_similarity_df.to_csv(save_path, index=False)
        print(f"[INFO] {idx + 1}번째까지 저장 완료: {save_path}")
        
        outputs = []
        indices = []

if outputs:
    for i, pred in zip(indices, outputs):
        low_similarity_df.at[i, "Cluster_refine"] = pred

    low_similarity_df.to_csv(save_path, index=False)
    print(f"[INFO] 최종 저장 완료 (총 {len(low_similarity_df)}개): {save_path}")
