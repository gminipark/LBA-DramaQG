import pandas as pd
low_similarity_df = pd.read_csv("./output/cluster_refine_result.csv")
df = pd.read_csv("./output/clustering_result.csv")

relation_col = "predicted_relation"
gt_col = "relation"
ref_col = "Cluster_refine"
ok_token = "Correct classification"

to_remove_mask = low_similarity_df[ref_col].astype(str).str.strip() != ok_token
to_remove = low_similarity_df.loc[to_remove_mask].copy()

if "cluster_name" in to_remove.columns:
    print("=== 클러스터별 삭제 대상 개수 ===")
    print(to_remove["cluster_name"].value_counts())

if "id" not in df.columns or "id" not in to_remove.columns:
    raise ValueError("df와 low_similarity_df 모두에 'id' 열이 필요합니다.")

ids_to_drop = set(to_remove["id"].tolist())
before_n = len(df)
df = df[~df["id"].isin(ids_to_drop)].reset_index(drop=True)
after_n = len(df)
print(f"\n삭제된 샘플 수: {before_n - after_n} / 남은 샘플 수: {after_n}")

cluster_purities = {}
total_correct = 0
total_samples = len(df)

print("\n=== Cluster-wise purity ===")
for pred_label, group in df.groupby(relation_col):
    counts = group[gt_col].value_counts()
    max_count = counts.max() if not counts.empty else 0
    purity = (max_count / len(group)) if len(group) > 0 else 0.0
    cluster_purities[pred_label] = purity
    total_correct += max_count
    print(f"{str(pred_label):30s}: {purity:.4f}  (n={len(group)})")

# 4) 전체 purity (가중 평균)
overall_purity = (total_correct / total_samples) if total_samples > 0 else 0.0
print("\n=== Overall purity ===")
print(f"{overall_purity:.4f}")

