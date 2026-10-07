from pathlib import Path
import pandas as pd, numpy as np, json
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results/baseline_rebuild_v010_20260922'
main_oof=ROOT/'results/raw_rebuild_snv_ladder_direct_20260916/tricor_moe/seed_1002/oof_predictions.csv'
main_fold=ROOT/'results/raw_rebuild_snv_ladder_direct_20260916/tricor_moe/seed_1002/fold_metrics.json'

methods={
'RCCR-MoE':(main_oof,main_fold),
'Elastic Net':(RES/'classical/elastic_net/oof_predictions.csv',RES/'classical/elastic_net/fold_metrics.csv'),
'Random Forest':(RES/'classical/random_forest/oof_predictions.csv',RES/'classical/random_forest/fold_metrics.csv'),
'XGBoost':(RES/'classical/xgboost/oof_predictions.csv',RES/'classical/xgboost/fold_metrics.csv'),
'Early MLP':(RES/'deep/early_mlp/oof_predictions.csv',RES/'deep/early_mlp/fold_metrics.csv'),
'Late MLP':(RES/'deep/late_mlp/oof_predictions.csv',RES/'deep/late_mlp/fold_metrics.csv'),
'GMU adapted':(RES/'deep/gmu_adapted/oof_predictions.csv',RES/'deep/gmu_adapted/fold_metrics.csv'),
'TFN adapted':(RES/'deep/tfn_adapted/oof_predictions.csv',RES/'deep/tfn_adapted/fold_metrics.csv'),
'LMF adapted':(RES/'deep/lmf_adapted/oof_predictions.csv',RES/'deep/lmf_adapted/fold_metrics.csv'),
'MulT adapted':(RES/'deep/mult_adapted/oof_predictions.csv',RES/'deep/mult_adapted/fold_metrics.csv'),
'MOGONET':(RES/'multiomics/mogonet/oof_predictions.csv',RES/'multiomics/mogonet/fold_metrics.csv'),
'DIABLO':(RES/'multiomics/diablo/oof_predictions.csv',RES/'multiomics/diablo/fold_metrics.csv'),
'Flexynesis':(RES/'multiomics/flexynesis/oof_predictions.csv',RES/'multiomics/flexynesis/fold_metrics.csv'),
}
def load_fold(p):
    if p.suffix=='.json':
        j=json.load(open(p))
        if isinstance(j,list):
            if j and isinstance(j[0],dict) and 'full' in j[0]:
                return pd.DataFrame([x['full'] for x in j])
            return pd.DataFrame(j)
        if 'folds' in j:
            return pd.DataFrame(j['folds'])
        return pd.DataFrame()
    return pd.read_csv(p)

main=pd.read_csv(main_oof).sort_values('subject_id')
assert main.subject_id.nunique()==139
y=main.true_label.to_numpy(); pmain=main.predicted_probability.to_numpy()
auc_main=roc_auc_score(y,pmain)
rng=np.random.default_rng(1002)
pos=np.where(y==1)[0]; neg=np.where(y==0)[0]
B=10000
boot_idx=[]
for b in range(B):
    ii=np.concatenate([rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)])
    boot_idx.append(ii)
boot_idx=np.array(boot_idx,dtype=np.int32)

rows=[]; raw_p=[]; comp_names=[]
for name,(op,fp) in methods.items():
    if not Path(op).exists(): continue
    o=pd.read_csv(op).sort_values('subject_id')
    assert np.array_equal(o.subject_id.to_numpy(),main.subject_id.to_numpy()), name
    assert np.array_equal(o.true_label.to_numpy(),y), name
    prob=o.predicted_probability.to_numpy()
    auc=roc_auc_score(y,prob)
    ap=average_precision_score(y,prob)
    folds=load_fold(Path(fp))
    # normalize main column names if needed
    fmap={'auc':'auc','auprc':'auprc','acc':'acc','f1':'f1','specificity':'specificity'}
    vals={}
    for k in fmap:
        col=fmap[k]
        if col in folds.columns:
            a=folds[col].astype(float).to_numpy()
        elif 'full_'+col in folds.columns:
            a=folds['full_'+col].astype(float).to_numpy()
        else:
            a=np.array([np.nan])
        vals[k]=(float(np.nanmean(a)),float(np.nanstd(a,ddof=1)) if np.isfinite(a).sum()>1 else np.nan)
    delta=auc_main-auc
    if name=='RCCR-MoE':
        ci=(0.0,0.0); pv=np.nan
    else:
        bd=np.empty(B,float)
        for b,ii in enumerate(boot_idx):
            yy=y[ii]
            bd[b]=roc_auc_score(yy,pmain[ii])-roc_auc_score(yy,prob[ii])
        ci=(float(np.percentile(bd,2.5)),float(np.percentile(bd,97.5)))
        nle=np.sum(bd<=0); nge=np.sum(bd>=0)
        pv=min(1.0,2*min((nle+1)/(B+1),(nge+1)/(B+1)))
        raw_p.append(pv); comp_names.append(name)
    rows.append({
        'method':name,
        'roc_auc_fold_mean':vals['auc'][0],'roc_auc_fold_sd':vals['auc'][1],
        'auprc_fold_mean':vals['auprc'][0],'auprc_fold_sd':vals['auprc'][1],
        'accuracy_fold_mean':vals['acc'][0],'accuracy_fold_sd':vals['acc'][1],
        'f1_fold_mean':vals['f1'][0],'f1_fold_sd':vals['f1'][1],
        'specificity_fold_mean':vals['specificity'][0],'specificity_fold_sd':vals['specificity'][1],
        'oof_roc_auc':auc,'oof_auprc':ap,
        'delta_auc_rccr_minus_method':delta,
        'delta_auc_ci95_low':ci[0],'delta_auc_ci95_high':ci[1],
        'paired_bootstrap_p_raw':pv,
        'bootstrap_replicates':B,
    })
if raw_p:
    m=len(raw_p)
    order=np.argsort(raw_p)
    adj=np.empty(m,float)
    running=0.0
    for rank,idx in enumerate(order):
        val=(m-rank)*raw_p[idx]
        running=max(running,val)
        adj[idx]=min(1.0,running)
    amap=dict(zip(comp_names,adj))
    for r in rows:r['p_holm']=amap.get(r['method'],np.nan)
df=pd.DataFrame(rows)
df.to_csv(RES/'main_comparison_interim.csv',index=False)
# pretty table
fmt=[]
for _,r in df.iterrows():
    fmt.append({
      'Method':r.method,
      'ROC-AUC':f"{r.roc_auc_fold_mean:.4f}±{r.roc_auc_fold_sd:.4f}",
      'AUPRC':f"{r.auprc_fold_mean:.4f}±{r.auprc_fold_sd:.4f}",
      'Accuracy':f"{r.accuracy_fold_mean:.4f}±{r.accuracy_fold_sd:.4f}",
      'F1':f"{r.f1_fold_mean:.4f}±{r.f1_fold_sd:.4f}",
      'Specificity':f"{r.specificity_fold_mean:.4f}±{r.specificity_fold_sd:.4f}",
      'OOF AUC':f"{r.oof_roc_auc:.4f}",
      'ΔAUC RCCR-method':f"{r.delta_auc_rccr_minus_method:.4f}",
      '95% CI':f"[{r.delta_auc_ci95_low:.4f}, {r.delta_auc_ci95_high:.4f}]" if r.method!='RCCR-MoE' else '—',
      'p raw':f"{r.paired_bootstrap_p_raw:.4g}" if r.method!='RCCR-MoE' else '—',
      'p Holm':f"{r.p_holm:.4g}" if r.method!='RCCR-MoE' else '—',
    })
pd.DataFrame(fmt).to_csv(RES/'main_comparison_interim_pretty.csv',index=False)
print(pd.DataFrame(fmt).to_string(index=False))
