"""Export training-only selected/scaled folds for the original DIABLO runner."""
from pathlib import Path
import sys,argparse
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from raw_rebuild import load_raw_modalities,fit_fold_selection,materialize_selected,fit_fold_scalers,apply_fold_scalers
from data import make_split,AlignedModalities
from train_raw_rebuild_sersnorm_20260916 import apply_sers_sample_norm
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,default=ROOT/'data/raw_rebuild_20260916');p.add_argument('--out-dir',type=Path,default=ROOT/'data/diablo_folds');a=p.parse_args()
    if a.out_dir.exists() and any(a.out_dir.iterdir()):raise SystemExit('Refusing nonempty output directory')
    raw=load_raw_modalities(a.data_dir);dummy=AlignedModalities(raw.sers,raw.tran_fpkm,raw.meta,raw.labels,raw.ids,raw.sers_names,raw.tran_names,raw.meta_names)
    for fold in range(5):
        tr,va,te=make_split(dummy,1002,fold,5);sel=fit_fold_selection(raw,tr,tran_k=512,meta_k=128)
        ds=[apply_sers_sample_norm(materialize_selected(raw,idx,sel),'snv') for idx in [tr,va,te]]
        scalers=fit_fold_scalers(ds[0]);fd=a.out_dir/f'fold_{fold}';fd.mkdir(parents=True)
        for part,data in zip(['train','val','test'],ds):
            data=apply_fold_scalers(data,scalers)
            for mod in ['sers','tran','meta']:pd.DataFrame(getattr(data,mod)).to_csv(fd/f'{part}_{mod}.csv',index=False)
            pd.DataFrame({'subject_id':data.ids,'y':data.labels}).to_csv(fd/f'{part}_labels.csv',index=False)
