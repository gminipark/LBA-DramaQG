import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

classification_result = pd.read_csv('./output/embedding_result.csv')

novel_df = classification_result[classification_result['classification_label'] == 'novel']

df = novel_df.copy()                 
relation_col = "predicted_relation"
emb_col = "embedding"             
target_cluster_count = 6            
pca_components = 50
max_iter = 50
tol = 1e-4

if emb_col not in df.columns or relation_col not in df.columns:
    raise ValueError(f"DataFrame에 '{emb_col}', '{relation_col}' 열이 필요")

X = np.vstack([np.asarray(v, dtype=np.float64) for v in df[emb_col].tolist()])  
N, d = X.shape
print(f"총 샘플 수 유지: {N} (행 삭제 없음)")

iteration = 1
while True:
    # 현재 relation 분포
    vc = df[relation_col].value_counts()
    K_curr = len(vc)
    print(f"\n=== Iter {iteration} ===")
    print(f"현재 relation 개수: {K_curr}")
    if K_curr <= target_cluster_count:
        print(f"목표 도달: relation 개수 {K_curr} ≤ {target_cluster_count}")
        break

    min_count = vc.min()
    rare_rels = set(vc[vc == min_count].index)
    print(f"희소(rare) relation → {sorted(rare_rels)} (count={min_count})")

    is_unlabeled = df[relation_col].isin(rare_rels).values
    is_labeled   = ~is_unlabeled

    labeled_relations = sorted(df.loc[is_labeled, relation_col].unique())
    K = len(labeled_relations)
    if K == 0:
        raise RuntimeError("모든 relation이 rare로 선택됨")
    print(f"재할당 기준으로 사용할 known relations({K}): {labeled_relations}")

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=min(pca_components, X_scaled.shape[1]-1), svd_solver="auto", random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    X_norm = normalize(X_pca, norm="l2", axis=1)

    rel2idx = {r: i for i, r in enumerate(labeled_relations)}
    idx2rel = {i: r for r, i in rel2idx.items()}

    centroids = np.zeros((K, X_norm.shape[1]), dtype=np.float64)
    for k, rel in enumerate(labeled_relations):
        centroids[k] = X_norm[(df[relation_col].values == rel) & is_labeled].mean(axis=0)

    labels = np.full(N, -1, dtype=int)
    labels[is_labeled] = df.loc[is_labeled, relation_col].map(rel2idx).values

    if is_unlabeled.any():
        dist0 = cdist(X_norm[is_unlabeled], centroids, metric="euclidean")
        labels[is_unlabeled] = dist0.argmin(axis=1)

    for _ in range(max_iter):
        prev = centroids.copy()
        for k in range(K):
            mask_l = is_labeled & (labels == k)  
            mask_u = is_unlabeled & (labels == k)
            if mask_l.any() or mask_u.any():
                centroids[k] = X_norm[mask_l | mask_u].mean(axis=0)
            else:
                centroids[k] = prev[k]
        if is_unlabeled.any():
            dist = cdist(X_norm[is_unlabeled], centroids, metric="euclidean")
            labels[is_unlabeled] = dist.argmin(axis=1)
        shift = np.linalg.norm(centroids - prev) / (np.linalg.norm(prev) + 1e-12)
        if shift < tol:
            break


    if is_unlabeled.any():
        assigned_rel = np.array([idx2rel[i] for i in labels[is_unlabeled]], dtype=object)
        df.loc[is_unlabeled, relation_col] = assigned_rel

    vc_after = df[relation_col].value_counts()
    print("재할당 후 relation 분포(상위 10개):")
    print(vc_after.head(10))

    iteration += 1

print("\n=== 최종 결과 ===")
print("최종 relation 개수:", df[relation_col].nunique())
print("최종 샘플 수(행):", len(df))

final_relations = sorted(df[relation_col].unique())
print(f"최종 relation 그룹 (총 {len(final_relations)}개): {final_relations}\n")

for rel in final_relations:
    print(f"\n===== 최종 클러스터: {rel} =====")
    cluster_df = df[df[relation_col] == rel]  
    vc = cluster_df['relation'].value_counts()  
    print(vc)

df.to_csv('./output/clustering_result.csv', index=False)
