from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, accuracy_score
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/raw_rebuild_simple_incremental_validation_v010_20260916'
if OUT.exists(): raise RuntimeError(f'Refusing to overwrite {OUT}')
OUT.mkdir(parents=True)
SEED=1002; B=10000
names={
'snv':{'sers':'raw_rebuild_snv_simple_subset_sers_20260916','tran':'raw_rebuild_snv_simple_subset_tran_20260916','meta':'raw_rebuild_snv_simple_subset_meta_20260916','sers_tran':'raw_rebuild_snv_simple_subset_sers_tran_20260916','sers_meta':'raw_rebuild_snv_simple_subset_sers_meta_20260916','tran_meta':'raw_rebuild_snv_simple_subset_tran_meta_retry1_20260916','all3':'raw_rebuild_snv_ablation_no_shared_private_20260916'},
'l2':{'sers':'raw_rebuild_l2_simple_subset_sers_20260916','tran':'raw_rebuild_l2_simple_subset_tran_retry1_20260916','meta':'raw_rebuild_l2_simple_subset_meta_retry1_20260916','sers_tran':'raw_rebuild_l2_simple_subset_sers_tran_retry1_20260916','sers_meta':'raw_rebuild_l2_simple_subset_sers_meta_retry1_20260916','tran_meta':'raw_rebuild_l2_simple_subset_tran_meta_retry1_20260916','all3':'raw_rebuild_l2_ablation_no_shared_private_20260916'}}
def load(norm,subset):
 p=ROOT/'results'/names[norm][subset]/'tricor_moe/seed_1002/oof_predictions.csv'
 return pd.read_csv(p).sort_values('subject_id').reset_index(drop=True)
def fm(mode,subset,d):
 rows=[]
 for f,g in d.groupby('fold',sort=True):
  y=g.true_label.to_numpy(int); p=g.predicted_probability.to_numpy(float)
  rows.append({'mode':mode,'subset':subset,'fold':int(f),'auc':roc_auc_score(y,p),'ap':average_precision_score(y,p),'brier':brier_score_loss(y,p),'acc':accuracy_score(y,g.predicted_label)})
 return rows
D={}; fr=[]
for mode in ['snv','l2']:
 for subset in names[mode]:
  d=load(mode,subset); D[(mode,subset)]=d; fr+=fm(mode,subset,d)
folds=pd.DataFrame(fr); folds.to_csv(OUT/'fold_metrics.csv',index=False)
summary=folds.groupby(['mode','subset']).agg(auc_mean=('auc','mean'),auc_std=('auc','std'),ap_mean=('ap','mean'),brier_mean=('brier','mean'),acc_mean=('acc','mean')).reset_index(); summary.to_csv(OUT/'subset_summary_foldwise.csv',index=False)
def boot_delta(da,db,seed):
 assert np.array_equal(da.subject_id,db.subject_id) and np.array_equal(da.true_label,db.true_label) and np.array_equal(da.fold,db.fold)
 rng=np.random.default_rng(seed); per=[]; obs=[]
 for f in sorted(da.fold.unique()):
  a=da[da.fold==f]; b=db[db.fold==f]; y=a.true_label.to_numpy(int); pa=a.predicted_probability.to_numpy(float); pb=b.predicted_probability.to_numpy(float)
  pos=np.where(y==1)[0]; neg=np.where(y==0)[0]; obs.append(roc_auc_score(y,pa)-roc_auc_score(y,pb))
  ip=rng.integers(0,len(pos),size=(B,len(pos))); ing=rng.integers(0,len(neg),size=(B,len(neg)))
  def ba(p):
   pp=p[pos][ip]; pn=p[neg][ing]
   return ((pp[:,:,None]>pn[:,None,:]).mean((1,2))+.5*(pp[:,:,None]==pn[:,None,:]).mean((1,2)))
  per.append(ba(pa)-ba(pb))
 vals=np.mean(np.vstack(per),axis=0); delta=float(np.mean(obs)); lo,hi=np.quantile(vals,[.025,.975]); p2=min(1.0,2*min((np.sum(vals<=0)+1)/(B+1),(np.sum(vals>=0)+1)/(B+1)))
 return delta,float(lo),float(hi),float(p2)
comps=[('sers_tran','sers','add_transcript_to_sers'),('sers_meta','sers','add_meta_to_sers'),('sers_tran','tran','add_sers_to_transcript'),('tran_meta','tran','add_meta_to_transcript'),('all3','sers_tran','add_meta_to_sers_tran'),('all3','sers_meta','add_transcript_to_sers_meta'),('all3','tran_meta','add_sers_to_tran_meta')]
rows=[]
for mi,mode in enumerate(['snv','l2']):
 for ci,(a,b,label) in enumerate(comps):
  d,lo,hi,p=boot_delta(D[(mode,a)],D[(mode,b)],SEED+100*mi+ci); rows.append({'mode':mode,'comparison':label,'model_a':a,'model_b':b,'delta_auc_meanfold_a_minus_b':d,'ci_low':lo,'ci_high':hi,'p_two_sided':p})
pd.DataFrame(rows).to_csv(OUT/'incremental_paired_bootstrap.csv',index=False)
stab=[]
for i,subset in enumerate(names['snv']):
 a=D[('l2',subset)]; b=D[('snv',subset)]; d,lo,hi,p=boot_delta(a,b,SEED+500+i)
 cor=[]
 for f in sorted(a.fold.unique()):
  aa=a[a.fold==f]; bb=b[b.fold==f]; cor.append(spearmanr(aa.predicted_probability,bb.predicted_probability).statistic)
 stab.append({'subset':subset,'delta_l2_minus_snv':d,'ci_low':lo,'ci_high':hi,'p_two_sided':p,'mean_withinfold_spearman':float(np.nanmean(cor)),'label_agreement':float(np.mean(a.predicted_label.to_numpy(int)==b.predicted_label.to_numpy(int)))})
pd.DataFrame(stab).to_csv(OUT/'snv_vs_l2_stability.csv',index=False)

def load_path(p): return pd.read_csv(p).sort_values('subject_id').reset_index(drop=True)
arch={
 'raw':(ROOT/'results/raw_rebuild_ablation_20260916/no_shared_private/tricor_moe/seed_1002/oof_predictions.csv',ROOT/'results/raw_rebuild_confirmatory_20260916/tricor_moe/seed_1002/oof_predictions.csv'),
 'snv':(ROOT/'results/raw_rebuild_snv_ablation_no_shared_private_20260916/tricor_moe/seed_1002/oof_predictions.csv',ROOT/'results/raw_rebuild_snv_ablation_full_20260916/tricor_moe/seed_1002/oof_predictions.csv'),
 'l2':(ROOT/'results/raw_rebuild_l2_ablation_no_shared_private_20260916/tricor_moe/seed_1002/oof_predictions.csv',ROOT/'results/raw_rebuild_l2_ablation_full_20260916/tricor_moe/seed_1002/oof_predictions.csv')}
ar=[]
for i,(mode,(sp,fp)) in enumerate(arch.items()):
 sd,fd=load_path(sp),load_path(fp); d,lo,hi,p=boot_delta(sd,fd,SEED+900+i)
 ar.append({'mode':mode,'delta_simple_minus_full_meanfold_auc':d,'ci_low':lo,'ci_high':hi,'p_two_sided':p})
pd.DataFrame(ar).to_csv(OUT/'simple_vs_full_architecture.csv',index=False)
report={'seed':SEED,'bootstrap_replicates':B,'primary_metric':'mean of 5 held-out-fold AUCs','bootstrap':'paired class-stratified bootstrap within held-out fold, then mean fold delta','architecture_simple':'TriCor-MoE with --no-shared-private; all other locked hyperparameters unchanged','guardrail':'No split, feature-selection, seed or locked-hyperparameter changes.'}
(OUT/'summary.json').write_text(json.dumps(report,indent=2))
print(summary.to_string(index=False)); print('\nINCREMENTAL'); print(pd.DataFrame(rows).to_string(index=False)); print('\nSTABILITY'); print(pd.DataFrame(stab).to_string(index=False)); print('\nARCH'); print(pd.DataFrame(ar).to_string(index=False))
