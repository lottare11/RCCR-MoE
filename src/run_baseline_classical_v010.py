from pathlib import Path
import json, csv, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from raw_rebuild import load_raw_modalities, fit_fold_selection, materialize_selected, fit_fold_scalers, apply_fold_scalers
from data import make_split, AlignedModalities
from train import binary_metrics, select_threshold

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'baseline_rebuild_v010_20260922'/'classical'
OUT.mkdir(parents=True,exist_ok=True)
raw=load_raw_modalities(ROOT/'data'/'raw_rebuild_20260916')
dummy=AlignedModalities(raw.sers,raw.tran_fpkm,raw.meta,raw.labels,raw.ids,raw.sers_names,raw.tran_names,raw.meta_names)

grids={
'elastic_net':[
 {'C':0.1,'l1_ratio':0.2},{'C':0.1,'l1_ratio':0.5},{'C':1.0,'l1_ratio':0.2},{'C':1.0,'l1_ratio':0.5},{'C':10.0,'l1_ratio':0.5}],
'random_forest':[
 {'n_estimators':500,'max_features':'sqrt','min_samples_leaf':1},
 {'n_estimators':500,'max_features':0.2,'min_samples_leaf':1},
 {'n_estimators':800,'max_features':'sqrt','min_samples_leaf':2}],
'xgboost':[
 {'n_estimators':300,'max_depth':2,'learning_rate':0.03,'subsample':0.8,'colsample_bytree':0.8},
 {'n_estimators':300,'max_depth':3,'learning_rate':0.03,'subsample':0.8,'colsample_bytree':0.8},
 {'n_estimators':500,'max_depth':2,'learning_rate':0.02,'subsample':0.9,'colsample_bytree':0.8}],
}

def make_model(name,p):
    if name=='elastic_net':
        return LogisticRegression(max_iter=5000,class_weight='balanced',solver='saga',penalty='elasticnet',random_state=1002,**p)
    if name=='random_forest':
        return RandomForestClassifier(class_weight='balanced',random_state=1002,n_jobs=-1,**p)
    if name=='xgboost':
        return XGBClassifier(objective='binary:logistic',eval_metric='logloss',random_state=1002,n_jobs=8,tree_method='hist',**p)
    raise KeyError(name)

for name,grid in grids.items():
    od=OUT/name; od.mkdir(parents=True,exist_ok=True)
    folds=[]; oof=[]; metas=[]
    for fold in range(5):
        tri,vi,tei=make_split(dummy,1002,fold,5)
        sel=fit_fold_selection(raw,tri,tran_k=512,meta_k=128)
        tr=materialize_selected(raw,tri,sel); va=materialize_selected(raw,vi,sel); te=materialize_selected(raw,tei,sel)
        for ds in (tr,va,te):
            mu=ds.sers.mean(axis=1,keepdims=True); sd=ds.sers.std(axis=1,keepdims=True)
            ds.sers=((ds.sers-mu)/np.maximum(sd,1e-8)).astype(np.float32)
        sc=fit_fold_scalers(tr); tr=apply_fold_scalers(tr,sc); va=apply_fold_scalers(va,sc); te=apply_fold_scalers(te,sc)
        Xtr=np.concatenate([tr.sers,tr.tran,tr.meta],1); Xva=np.concatenate([va.sers,va.tran,va.meta],1); Xte=np.concatenate([te.sers,te.tran,te.meta],1)
        best=None; best_auc=-1
        for p in grid:
            m=make_model(name,p); m.fit(Xtr,tr.labels); pv=m.predict_proba(Xva)[:,1]; a=roc_auc_score(va.labels,pv)
            if a>best_auc: best_auc=float(a); best=(p,m,pv)
        p,m,pv=best
        thr=select_threshold(va.labels,pv,objective='balanced_acc')
        pt=m.predict_proba(Xte)[:,1]; met=binary_metrics(te.labels,pt,thr); met.update({'fold':fold,'best_val_auc':best_auc,'selected_params':json.dumps(p,sort_keys=True)})
        folds.append(met)
        yp=(pt>=thr).astype(int)
        for sid,y,prob,pred in zip(te.ids,te.labels,pt,yp):
            oof.append({'method':name,'subject_id':int(sid),'fold':fold,'true_label':int(y),'predicted_probability':float(prob),'predicted_label':int(pred),'threshold':float(thr)})
        metas.append({'fold':fold,'train_size':len(tr.ids),'val_size':len(va.ids),'test_size':len(te.ids),'tran_selected_count':512,'meta_selected_count':128,'sers_selected_count':1796,'selected_params':json.dumps(p,sort_keys=True)})
        print(name,fold,met,flush=True)
    pd.DataFrame(folds).to_csv(od/'fold_metrics.csv',index=False)
    pd.DataFrame(oof).sort_values(['fold','subject_id']).to_csv(od/'oof_predictions.csv',index=False)
    pd.DataFrame(metas).to_csv(od/'fold_metadata.csv',index=False)
    summ={}
    for k in ['auc','auprc','acc','f1','specificity','brier']:
        a=np.array([r[k] for r in folds],float); summ[k+'_mean']=a.mean(); summ[k+'_std']=a.std(ddof=1)
    (od/'summary.json').write_text(json.dumps({'summary':summ,'seed':1002,'n':139,'pipeline':'raw_rebuild SNV; train-only 512 transcript/128 metabolite selection; validation hyperparameter and threshold selection'},indent=2))
