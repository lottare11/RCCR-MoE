"""Meaningful smoke checks for fold isolation and exact decision decomposition."""
from pathlib import Path
import sys,tempfile,copy
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from make_synthetic_data import generate
from raw_rebuild import load_raw_modalities,fit_fold_selection,materialize_selected,fit_fold_scalers,apply_fold_scalers
from data import build_feature_group_spec
from model import build_tricor_moe
torch.set_num_threads(2)
with tempfile.TemporaryDirectory() as td:
    generate(Path(td)/'data');r=load_raw_modalities(Path(td)/'data');idx=np.arange(24)
    a=fit_fold_selection(r,idx,tran_k=48,meta_k=24,tran_prevar_k=128)
    other=copy.copy(r);other.labels=r.labels.copy();other.labels[24:]=1-other.labels[24:]
    other.tran_fpkm=np.asarray(r.tran_fpkm).copy();other.meta=np.asarray(r.meta).copy()
    other.tran_fpkm[24:]=1e6;other.meta[24:]=1e6
    b=fit_fold_selection(other,idx,tran_k=48,meta_k=24,tran_prevar_k=128)
    assert np.array_equal(a.tran_indices,b.tran_indices)
    assert np.array_equal(a.meta_indices,b.meta_indices)
    d=materialize_selected(r,idx,a);gs=build_feature_group_spec(d,seed=1002,sers_groups=8,tran_groups=4,meta_groups=4)
    d=apply_fold_scalers(d,fit_fold_scalers(d))
    model=build_tricor_moe(gs,d_model=32,dropout=.1,use_shared_private=False).eval()
    x=[torch.tensor(getattr(d,k)[:4]) for k in ['sers','tran','meta']]
    with torch.no_grad():o=model(*x)
    margin=o['logits'][:,1]-o['logits'][:,0]
    experts=.5*o['gate']*(o['expert_logits'][:,:,1]-o['expert_logits'][:,:,0])
    reconstructed=experts.sum(1)+.5*(o['global_logits'][:,1]-o['global_logits'][:,0])+o['refinement_logits'][:,1]-o['refinement_logits'][:,0]
    torch.testing.assert_close(margin,reconstructed,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(o['gate'].sum(1),torch.ones(4))
    mask=torch.tensor([[1.,0.,1.]]).expand(4,-1)
    with torch.no_grad():masked=model(*x,modality_mask=mask)
    assert torch.equal(masked['gate'][:,1],torch.zeros(4))
print('PASS: held-out changes leave feature selection unchanged; contributions reconstruct margin; unavailable expert has zero gate.')
