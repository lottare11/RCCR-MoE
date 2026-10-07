"""Summarize the original DIABLO runner's held-out predictions by fold."""
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,f1_score,recall_score
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'results/baseline_rebuild_v010_20260922/multiomics/diablo'
d=pd.read_csv(p/'oof_predictions.csv');rows=[]
for fold,g in d.groupby('fold'):
    y=g.true_label;prob=g.predicted_probability;pred=g.predicted_label
    rows.append(dict(fold=int(fold),auc=roc_auc_score(y,prob),auprc=average_precision_score(y,prob),
                     acc=accuracy_score(y,pred),f1=f1_score(y,pred),specificity=recall_score(y,pred,pos_label=0)))
pd.DataFrame(rows).to_csv(p/'fold_metrics.csv',index=False)
