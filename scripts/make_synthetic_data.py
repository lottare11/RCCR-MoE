"""Create artificial inputs for execution testing, never patient substitutes."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
def generate(out,n=40,seed=721):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Output directory must be empty')
    out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(seed)
    y=np.tile([0,1],n//2);rng.shuffle(y)
    wave=np.linspace(30,4300,1934,dtype=np.float32)
    s=rng.lognormal(1,.2,(n,len(wave))).astype('float32')
    t=rng.lognormal(1,.7,(n,256)).astype('float32')
    m=rng.lognormal(1,.5,(n,64)).astype('float32')
    s[:,100:120]+=y[:,None]*.4;t[:,:12]+=y[:,None]*1.5;m[:,:4]+=y[:,None]*.6
    pd.DataFrame({'ID':np.arange(1,n+1),'label':y}).to_csv(out/'canonical_subjects.csv',index=False)
    np.save(out/'sers_mean3_139x1934.npy',s)
    np.save(out/'transcript_fpkm_139x62354.npy',t)
    np.save(out/'transcript_count_139x62354.npy',np.rint(t*10).astype('float32'))
    np.save(out/'metabolomics_full_139x2388.npy',m)
    np.savetxt(out/'sers_wavenumbers.csv',wave,delimiter=',')
    pd.DataFrame({'gene_id':[f'SYN_G{i:04}' for i in range(t.shape[1])],
                  'gene_name':[f'SYN_G{i:04}' for i in range(t.shape[1])]}).to_csv(out/'transcript_feature_metadata.csv',index=False)
    pd.DataFrame({'Compound_ID':[f'SYN_M{i:04}' for i in range(m.shape[1])]}).to_csv(out/'metabolomics_feature_metadata.csv',index=False)
    (out/'SYNTHETIC_ONLY.json').write_text(json.dumps({'synthetic':True,'seed':seed,'n':n,'note':'Historical filenames are required by the loader; actual shapes intentionally differ.'},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',required=True,type=Path);a=p.parse_args();generate(a.out_dir)
