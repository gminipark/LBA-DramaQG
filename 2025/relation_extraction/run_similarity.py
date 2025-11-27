import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from utils_for_similarity import embed_texts 


relation_col = "predicted_relation"
emb_col = "embedding"
top_percent_as_low_similarity = 0.80

df = pd.read_csv("./output/clustering_result.csv")

if emb_col not in df.columns or relation_col not in df.columns:
    raise ValueError(f"DataFrame에 '{emb_col}', '{relation_col}' 열이 필요합니다.")

unique_clusters = sorted(df[relation_col].unique())
print(f"최종 클러스터 개수: {len(unique_clusters)}, 라벨: {unique_clusters}")


def _to_vec(arr):
    v = np.asarray(arr, dtype=np.float64)
    return v


cluster_names = [str(c) for c in unique_clusters]
cluster_name_embs = embed_texts(cluster_names)

name2emb = {str(name): cluster_name_embs[i] for i, name in enumerate(unique_clusters)}

df["_vec"] = df[emb_col].apply(_to_vec)
dims = df["_vec"].apply(lambda v: v.shape[0]).unique()

if len(dims) != 1:
    raise ValueError(f"임베딩 차원이 여러 개입니다: {dims}")

d = int(dims[0])

rows = []
for cname in unique_clusters:
    mask = (df[relation_col] == cname).values
    if not mask.any():
        continue

    X = np.vstack(df.loc[mask, "_vec"].tolist())
    name_emb = name2emb[str(cname)].reshape(1, -1)

    if d == 768:
        dist = cdist(X, name_emb, metric="euclidean").ravel()
    else:
        X1, X2 = X[:, :768], X[:, 768:]
        d1 = cdist(X1, name_emb, metric="euclidean").ravel()
        d2 = cdist(X2, name_emb, metric="euclidean").ravel()
        dist = (d1 + d2) / 2.0

    n = len(dist)
    k = max(1, int(np.ceil(n * top_percent_as_low_similarity)))
    idx_top = np.argpartition(-dist, k - 1)[:k]
    idx_sorted = idx_top[np.argsort(-dist[idx_top])]

    sub = df.loc[mask].iloc[idx_sorted].copy()
    sub["cluster_name"] = cname
    sub["euclid_dist_to_name"] = dist[idx_sorted]
    if d == 1536:
        sub["euclid_dist_front768"] = d1[idx_sorted]
        sub["euclid_dist_back768"] = d2[idx_sorted]
    rows.append(sub)

low_similarity_df = pd.concat(rows, axis=0).reset_index(drop=True)

low_similarity_df = low_similarity_df.sort_values(
    by=["cluster_name", "euclid_dist_to_name"],
    ascending=[True, False]
).reset_index(drop=True)

low_similarity_df = low_similarity_df.drop(columns=["_vec"])

print(low_similarity_df["cluster_name"].value_counts())

display_cols = ["cluster_name", "euclid_dist_to_name", relation_col]
extra = []
if "tagged_text" in df.columns:
    extra.append("tagged_text")
preview_cols = display_cols + extra

low_similarity_df.to_csv("./output/low_similarity_df.csv", index=False)
