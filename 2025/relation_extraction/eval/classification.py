import pandas as pd
import re
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

INITIAL_DESC_CSV = "./relation_35_description_w_no_relation.csv"
preds_df = pd.read_csv('./output/no_rel_in_descriptions_result.csv')

def extract_relation_from_text(text):
    """
    continual_description 문자열에서 
    'relation name : org:stateorprovince_of_founding' 형식으로 되어 있는 경우,
    그 뒷부분만 추출하여 반환.
    """
    if pd.isna(text):
        return None
    
    match = re.search(r'relation name\s*[:\-]\s*([^\s,.\n]+)', text.strip(), re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return None  # 형식이 없는 경우 None 반환

def conditional_update(row):
    extracted = extract_relation_from_text(row['continual_description'])
    if extracted:  # relation name이 발견된 경우
        return extracted
    else:
        return row['predicted_relation']  # 기존 값 유지

preds_df['predicted_relation'] = preds_df.apply(conditional_update, axis=1)
output_path = './output/no_rel_in_descriptions_result.csv'
preds_df.to_csv(output_path, index=False)


initial_desc_df = pd.read_csv(INITIAL_DESC_CSV)
initial_relations = set(initial_desc_df["relation"].astype(str).str.strip())

df_eval = preds_df.copy()

df_eval = df_eval[df_eval["target"].notna() & df_eval["predicted_relation"].notna()].copy()

df_eval["pred_binary"] = df_eval["predicted_relation"].apply(lambda x: 0 if x in initial_relations else 1)

df_eval["target_int"] = df_eval["target"].astype(int)

y_true = df_eval["target_int"]
y_pred = df_eval["pred_binary"]

print("=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=['known(0)', 'novel(1)']))

cm = confusion_matrix(y_true, y_pred)

print("=== Confusion Matrix ===")
print(cm)

# 시각화
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=['Pred Known (0)', 'Pred Novel (1)'],
            yticklabels=['Actual Known (0)', 'Actual Novel (1)'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()


