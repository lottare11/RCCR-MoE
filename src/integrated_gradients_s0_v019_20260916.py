from __future__ import annotations
import argparse, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold, train_test_split
from model import build_tricor_moe
from raw_rebuild import load_raw_modalities, fit_fold_selection, materialize_selected, fit_fold_scalers, apply_fold_scalers
from train_raw_rebuild_sersnorm_20260916 import apply_sers_sample_norm

def load_model(path: Path, device):
    ck=torch.load(path,map_location=device); a=ck['args']; gs=SimpleNamespace(**ck['group_spec'])
    model=build_tricor_moe(gs,d_model=int(a['d_model']),dropout=float(a['dropout']),
        use_consensus_token=not bool(a.get('no_consensus_token',False)),
        use_conflict_token=not bool(a.get('no_conflict_token',False)),
        use_shared_private=not bool(a.get('no_shared_private',False)),
        use_moe_routing=not bool(a.get('no_moe_routing',False)),
        use_refinement_head=not bool(a.get('no_refinement_head',False))).to(device)
    model.load_state_dict(ck['state_dict']); model.eval(); return model,a
def integrated_gradients(model,data,device,steps=32):
    xs=[torch.tensor(data.sers,dtype=torch.float32,device=device),
        torch.tensor(data.tran,dtype=torch.float32,device=device),
        torch.tensor(data.meta,dtype=torch.float32,device=device)]
    acc=[torch.zeros_like(x) for x in xs]
    for k in range(steps):
        alpha=(k+0.5)/steps
        cur=[(x*alpha).detach().requires_grad_(True) for x in xs]
        out=model(cur[0],cur[1],cur[2])
        target=(out['logits'][:,1]-out['logits'][:,0]).sum()
        grads=torch.autograd.grad(target,cur,retain_graph=False,create_graph=False)
        for j,g in enumerate(grads): acc[j]+=g.detach()
    return [(xs[j]*acc[j]/steps).detach().cpu().numpy() for j in range(3)]

def stability(rows):
    df=pd.DataFrame(rows); out=[]
    for feature,sub in df.groupby('feature'):
        out.append({'feature':feature,'mean_abs_ig':sub.mean_abs_ig.mean(),
                    'sd_abs_ig':sub.mean_abs_ig.std(ddof=1),
                    'fold_occurrence':sub.fold.nunique()})
    return pd.DataFrame(out).sort_values('mean_abs_ig',ascending=False)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--raw-data-dir',type=Path,default=Path('data/raw_rebuild_20260916'))
    p.add_argument('--out-dir',type=Path,required=True); p.add_argument('--seed',type=int,default=1002)
    p.add_argument('--steps',type=int,default=32); p.add_argument('--gradx-dir',type=Path,default=None)
    args=p.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw=load_raw_modalities(args.raw_data_dir); model0,ta=load_model(args.run_dir/'fold_0_best.pt',device)
    if int(ta['seed'])!=args.seed: raise ValueError('seed mismatch')
    skf=StratifiedKFold(n_splits=int(ta['n_splits']),shuffle=True,random_state=args.seed)
    splits=list(skf.split(np.zeros_like(raw.labels),raw.labels)); all_rows={k:[] for k in ['sers','transcriptome','metabolome']}
    for fold,(tv,test_idx) in enumerate(splits):
        train_idx,val_idx=train_test_split(tv,test_size=.2,random_state=args.seed+fold,stratify=raw.labels[tv])
        sel=fit_fold_selection(raw,train_idx,tran_k=int(ta['tran_k']),meta_k=int(ta['meta_k']),
            tran_prevar_k=int(ta['tran_prevar_k']),expressed_fpkm=float(ta['expressed_fpkm']),expressed_fraction=float(ta['expressed_fraction']))
        tr=materialize_selected(raw,train_idx,sel); te=materialize_selected(raw,test_idx,sel)
        apply_sers_sample_norm(tr,str(ta.get('sers_sample_norm','none'))); apply_sers_sample_norm(te,str(ta.get('sers_sample_norm','none')))
        sc=fit_fold_scalers(tr); te=apply_fold_scalers(te,sc)
        model,_=load_model(args.run_dir/f'fold_{fold}_best.pt',device); attrs=integrated_gradients(model,te,device,args.steps)
        names=[te.sers_names,te.tran_names,te.meta_names]
        for mod,a,nm in zip(['sers','transcriptome','metabolome'],attrs,names):
            ma=np.abs(a).mean(axis=0)
            for j,name in enumerate(nm): all_rows[mod].append({'fold':fold,'feature':str(name),'mean_abs_ig':float(ma[j])})
    summaries={}
    for mod,rows in all_rows.items():
        pd.DataFrame(rows).to_csv(args.out_dir/f'{mod}_ig_by_fold.csv',index=False)
        st=stability(rows); st.to_csv(args.out_dir/f'{mod}_ig_stability.csv',index=False)
        summaries[mod]={'n_features_union':int(len(st)),'top20':st.head(20).feature.tolist()}
    if args.gradx_dir:
        compare=[]
        files={'sers':'sers_attribution_feature_oof.csv','transcriptome':'transcript_attribution_feature_oof.csv','metabolome':'metabolome_attribution_feature_oof.csv'}
        for mod,f in files.items():
            gx=pd.read_csv(args.gradx_dir/f).groupby('feature').abs_attribution.mean().rename('gradx')
            ig=stability(all_rows[mod]).set_index('feature').mean_abs_ig.rename('ig')
            z=pd.concat([gx,ig],axis=1,join='inner').dropna(); rho,pv=spearmanr(z.gradx,z.ig)
            k=min(50,len(z)); a=set(z.gradx.nlargest(k).index); b=set(z.ig.nlargest(k).index)
            compare.append({'modality':mod,'n_shared_features':len(z),'spearman_rho':float(rho),'spearman_p':float(pv),
                            'top50_jaccard':len(a&b)/len(a|b) if a|b else np.nan})
        pd.DataFrame(compare).to_csv(args.out_dir/'ig_vs_gradx_consistency.csv',index=False); summaries['consistency']=compare
    report={'seed':args.seed,'steps':args.steps,'device':str(device),'sers_sample_norm':str(ta.get('sers_sample_norm','none')),
            'no_shared_private':bool(ta.get('no_shared_private',False)),'summary':summaries}
    (args.out_dir/'summary.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
