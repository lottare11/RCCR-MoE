"""A small CPU-only end-to-end run on newly generated synthetic data."""
from pathlib import Path
import argparse,json,os
from make_synthetic_data import generate
from run_primary import run,ROOT
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work-dir',type=Path,default=ROOT/'work/demo');a=p.parse_args()
    work=a.work_dir.resolve()
    if work.exists() and any(work.iterdir()):raise SystemExit('Use an empty --work-dir')
    generate(work/'data')
    config=json.loads((ROOT/'configs/primary.json').read_text())
    config.update(n_splits=2,epochs=2,patience=2,d_model=32,tran_k=48,meta_k=24,
                  tran_prevar_k=128,sers_groups=8,tran_groups=4,meta_groups=4,batch_size=8)
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
    run(config,work/'data',work/'results')
    print('Synthetic execution check completed. These metrics are not manuscript results.')
