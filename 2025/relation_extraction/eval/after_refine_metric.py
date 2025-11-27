import pandas as pd
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score


df = pd.read_csv("./output/clustering_result.csv")
low_similarity_df = pd.read_csv("./output/Cluster_refine_result.csv")

relation_col = "predicted_relation" 
gt_col = "relation"
ref_col = "Cluster_refine"
ok_token = "Correct classification"

y_true = df[gt_col].astype(str).str.strip()
y_pred = df[relation_col].astype(str).str.strip()

mask = y_true.ne("") & y_pred.ne("") & y_true.notna() & y_pred.notna()
y_true = y_true[mask].reset_index(drop=True)
y_pred = y_pred[mask].reset_index(drop=True)

ct = pd.crosstab(y_pred, y_true)

row_index = {lab: i for i, lab in enumerate(ct.index)}
col_index = {lab: j for j, lab in enumerate(ct.columns)}

rows = y_pred.map(row_index).to_numpy()
cols = y_true.map(col_index).to_numpy()

ct_values = ct.to_numpy()
cluster_sizes = ct.sum(axis=1).to_numpy()
class_sizes = ct.sum(axis=0).to_numpy()

intersections = ct_values[rows, cols]

cluster_den = np.where(cluster_sizes[rows] == 0, 1, cluster_sizes[rows])
class_den = np.where(class_sizes[cols] == 0, 1, class_sizes[cols])

prec_i = intersections / cluster_den
rec_i = intersections / class_den

b3_p = prec_i.mean()
b3_r = rec_i.mean()
b3_f1 = 0 if (b3_p + b3_r) == 0 else (2 * b3_p * b3_r) / (b3_p + b3_r)

hom = homogeneity_score(y_true, y_pred)
comp = completeness_score(y_true, y_pred)
v = v_measure_score(y_true, y_pred)

ari = adjusted_rand_score(y_true, y_pred)

print("===== B³ (B-cubed) =====")
print(f"B³ Precision : {b3_p:.6f}")
print(f"B³ Recall    : {b3_r:.6f}")
print(f"B³ F1        : {b3_f1:.6f}")

print("\n===== V-measure =====")
print(f"Homogeneity  : {hom:.6f}")
print(f"Completeness : {comp:.6f}")
print(f"V-measure(F1): {v:.6f}")

print("\n===== ARI =====")
print(f"ARI : {ari:.6f}")
