import pandas as pd
import numpy as np

df = pd.read_csv("./output/clustering_result.csv")

relation_col = "predicted_relation"  # 예측 클러스터 열
gt_col = "relation"                  # ground-truth 열

cluster_purities = {}
total_correct = 0
total_samples = len(df)

for rel, group in df.groupby(relation_col):
    counts = group[gt_col].value_counts()
    max_count = counts.max()
    purity = max_count / len(group)
    cluster_purities[rel] = purity
    total_correct += max_count

# 각 클러스터별 purity 출력
print("=== Cluster-wise purity ===")
for rel, p in cluster_purities.items():
    print(f"{rel:30s}: {p:.4f}")

# 전체 purity (weighted mean)
overall_purity = total_correct / total_samples
print("\n=== Overall purity ===")
print(f"{overall_purity:.4f}")
