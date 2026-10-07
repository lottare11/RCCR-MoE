from pathlib import Path
import json, csv, random, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from raw_rebuild import load_raw_modalities, fit_fold_selection, materialize_selected, fit_fold_scalers, apply_fold_scalers
from data import make_split, AlignedModalities, build_feature_group_spec
from model import GroupTokenizer
from train import binary_metrics, select_threshold

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'baseline_rebuild_v010_20260922'/'deep'
OUT.mkdir(parents=True,exist_ok=True)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

class Enc(nn.Module):
    def __init__(self,inp,d=64,drop=.1):
        super().__init__(); self.net=nn.Sequential(nn.Linear(inp,d*2),nn.LayerNorm(d*2),nn.GELU(),nn.Dropout(drop),nn.Linear(d*2,d),nn.LayerNorm(d),nn.GELU())
    def forward(self,x): return self.net(x)

class EarlyMLP(nn.Module):
    def __init__(self,dims):
        super().__init__(); n=sum(dims); self.net=nn.Sequential(nn.Linear(n,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,2))
    def forward(self,s,t,m): return self.net(torch.cat([s,t,m],1))

class LateMLP(nn.Module):
    def __init__(self,dims):
        super().__init__(); self.heads=nn.ModuleList([nn.Sequential(Enc(d,64,.1),nn.Linear(64,2)) for d in dims])
    def forward(self,s,t,m): return torch.stack([h(x) for h,x in zip(self.heads,[s,t,m])],0).mean(0)

class GMU(nn.Module):
    def __init__(self,dims):
        super().__init__(); d=64; self.encs=nn.ModuleList([Enc(x,d,.1) for x in dims]); self.gate=nn.Sequential(nn.Linear(d*3,d),nn.GELU(),nn.Linear(d,3)); self.head=nn.Linear(d,2)
    def forward(self,s,t,m):
        hs=[e(x) for e,x in zip(self.encs,[s,t,m])]; g=torch.softmax(self.gate(torch.cat(hs,1)),1); f=sum(h*g[:,i:i+1] for i,h in enumerate(hs)); return self.head(f)

class TFN(nn.Module):
    def __init__(self,dims):
        super().__init__(); d=24; self.encs=nn.ModuleList([Enc(x,d,.1) for x in dims]); self.head=nn.Sequential(nn.Linear((d+1)**3,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,2))
    def forward(self,s,t,m):
        hs=[e(x) for e,x in zip(self.encs,[s,t,m])]; aug=[torch.cat([torch.ones(x.size(0),1,device=x.device),x],1) for x in hs]
        z=torch.einsum('bi,bj,bk->bijk',aug[0],aug[1],aug[2]).flatten(1); return self.head(z)

class LMF(nn.Module):
    def __init__(self,dims):
        super().__init__(); d=64; r=8; o=64
        self.encs=nn.ModuleList([Enc(x,d,.1) for x in dims])
        self.factors=nn.ParameterList([nn.Parameter(torch.randn(r,d+1,o)*0.02) for _ in range(3)])
        self.bias=nn.Parameter(torch.zeros(o)); self.head=nn.Sequential(nn.LayerNorm(o),nn.GELU(),nn.Dropout(.1),nn.Linear(o,2))
    def forward(self,s,t,m):
        hs=[e(x) for e,x in zip(self.encs,[s,t,m])]; aug=[torch.cat([torch.ones(x.size(0),1,device=x.device),x],1) for x in hs]
        proj=[torch.einsum('bi,rio->bro',x,f) for x,f in zip(aug,self.factors)]
        fused=(proj[0]*proj[1]*proj[2]).sum(1)+self.bias
        return self.head(fused)

class MulTAdapted(nn.Module):
    def __init__(self,dims,groups):
        super().__init__(); d=64
        self.toks=nn.ModuleList([GroupTokenizer(g,d,.1) for g in groups])
        self.attn=nn.ModuleList([nn.MultiheadAttention(d,4,dropout=.1,batch_first=True) for _ in range(3)])
        self.norm=nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        self.head=nn.Sequential(nn.Linear(d*3,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,2))
    def forward(self,s,t,m):
        xs=[tok(x) for tok,x in zip(self.toks,[s,t,m])]
        outs=[]
        for i in range(3):
            ctx=torch.cat([xs[j] for j in range(3) if j!=i],1)
            a,_=self.attn[i](xs[i],ctx,ctx,need_weights=False)
            outs.append(self.norm[i](xs[i]+a).mean(1))
        return self.head(torch.cat(outs,1))

def build(name,dims,groups):
    return {'early_mlp':EarlyMLP,'late_mlp':LateMLP,'gmu_adapted':GMU,'tfn_adapted':TFN,'lmf_adapted':LMF}.get(name,lambda d:MulTAdapted(d,groups))(dims)

def pred(model,ds):
    ld=DataLoader(TensorDataset(torch.tensor(ds.sers,dtype=torch.float32),torch.tensor(ds.tran,dtype=torch.float32),torch.tensor(ds.meta,dtype=torch.float32),torch.tensor(ds.labels,dtype=torch.long)),batch_size=64,shuffle=False)
    ys=[]; ps=[]; model.eval()
    with torch.no_grad():
        for s,t,m,y in ld:
            p=torch.softmax(model(s.to(DEVICE),t.to(DEVICE),m.to(DEVICE)),1)[:,1].cpu().numpy(); ys.append(y.numpy()); ps.append(p)
    return np.concatenate(ys),np.concatenate(ps)

raw=load_raw_modalities(ROOT/'data'/'raw_rebuild_20260916')
dummy=AlignedModalities(raw.sers,raw.tran_fpkm,raw.meta,raw.labels,raw.ids,raw.sers_names,raw.tran_names,raw.meta_names)
methods=['early_mlp','late_mlp','gmu_adapted','tfn_adapted','lmf_adapted','mult_adapted']
for method in methods:
    od=OUT/method; od.mkdir(parents=True,exist_ok=True); folds=[]; oof=[]; meta=[]
    for fold in range(5):
        seed_all(1002+fold)
        tri,vi,tei=make_split(dummy,1002,fold,5)
        sel=fit_fold_selection(raw,tri,tran_k=512,meta_k=128)
        tr=materialize_selected(raw,tri,sel); va=materialize_selected(raw,vi,sel); te=materialize_selected(raw,tei,sel)
        for ds in (tr,va,te):
            mu=ds.sers.mean(axis=1,keepdims=True); sd=ds.sers.std(axis=1,keepdims=True); ds.sers=((ds.sers-mu)/np.maximum(sd,1e-8)).astype(np.float32)
        sc=fit_fold_scalers(tr); tr=apply_fold_scalers(tr,sc); va=apply_fold_scalers(va,sc); te=apply_fold_scalers(te,sc)
        spec=build_feature_group_spec(tr,seed=1002+fold,sers_groups=16,tran_groups=24,meta_groups=8)
        groups=[spec.sers_groups,spec.tran_groups,spec.meta_groups]
        model=build(method,(1796,512,128),groups).to(DEVICE)
        counts=np.bincount(tr.labels,minlength=2).astype(np.float32); w=counts.sum()/np.maximum(counts,1); w=w/w.mean()
        crit=nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32,device=DEVICE))
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        ld=DataLoader(TensorDataset(torch.tensor(tr.sers,dtype=torch.float32),torch.tensor(tr.tran,dtype=torch.float32),torch.tensor(tr.meta,dtype=torch.float32),torch.tensor(tr.labels,dtype=torch.long)),batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(1002+fold))
        best=None; best_auc=-1; bad=0; best_epoch=-1
        for ep in range(200):
            model.train()
            for s,t,m,y in ld:
                opt.zero_grad(); logits=model(s.to(DEVICE),t.to(DEVICE),m.to(DEVICE)); loss=crit(logits,y.to(DEVICE)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            vy,vp=pred(model,va); a=roc_auc_score(vy,vp)
            if a>best_auc+1e-10:
                best_auc=float(a); best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0; best_epoch=ep
            else:
                bad+=1
                if bad>=25: break
        model.load_state_dict(best)
        vy,vp=pred(model,va); thr=select_threshold(vy,vp,'balanced_acc')
        ty,tp=pred(model,te); met=binary_metrics(ty,tp,thr); met.update({'fold':fold,'best_val_auc':best_auc,'best_epoch':best_epoch})
        folds.append(met); yp=(tp>=thr).astype(int)
        for sid,y,p,pr in zip(te.ids,ty,tp,yp):
            oof.append({'method':method,'subject_id':int(sid),'fold':fold,'true_label':int(y),'predicted_probability':float(p),'predicted_label':int(pr),'threshold':float(thr)})
        meta.append({'fold':fold,'train_size':len(tr.ids),'val_size':len(va.ids),'test_size':len(te.ids),'sers_dim':1796,'tran_dim':512,'meta_dim':128,'best_epoch':best_epoch,'best_val_auc':best_auc})
        print(method,fold,met,flush=True)
        del model; torch.cuda.empty_cache()
    pd.DataFrame(folds).to_csv(od/'fold_metrics.csv',index=False); pd.DataFrame(oof).sort_values(['fold','subject_id']).to_csv(od/'oof_predictions.csv',index=False); pd.DataFrame(meta).to_csv(od/'fold_metadata.csv',index=False)
    summ={}
    for k in ['auc','auprc','acc','f1','specificity','brier']:
        a=np.array([r[k] for r in folds]); summ[k+'_mean']=float(a.mean()); summ[k+'_std']=float(a.std(ddof=1))
    prov={'summary':summ,'seed':1002,'pipeline':'raw_rebuild SNV + train-only feature selection and module grouping','adaptation':{'early_mlp':'concatenate three modalities then MLP','late_mlp':'three modality-specific MLP classifiers; decision logits averaged','gmu_adapted':'GMU-inspired softmax gating over three modality embeddings','tfn_adapted':'TFN-inspired augmented three-way outer-product tensor fusion','lmf_adapted':'LMF-inspired rank-8 factorized three-way fusion','mult_adapted':'MulT-inspired group-token cross-modal attention; train-only transcript/metabolite groups'}[method]}
    (od/'summary.json').write_text(json.dumps(prov,indent=2))
