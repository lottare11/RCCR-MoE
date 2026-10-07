"""Reproduce the recorded SERS-only random-position controls."""
from pathlib import Path
import argparse,json,os,subprocess,sys
from run_primary import ROOT,command
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,default=ROOT/'data/raw_rebuild_20260916');a=p.parse_args()
    cfg=json.loads((ROOT/'configs/sers_control.json').read_text())
    for mode in ['sample_perm','circular_shift']:
        out=ROOT/'results/sers_controls'/mode
        if out.exists() and any(out.iterdir()):raise SystemExit(f'Refusing nonempty output directory: {out}')
        cmd=command(cfg,a.data_dir,out);cmd[1]=str(ROOT/'src/train_raw_rebuild_serssanity_v001_20260916.py')
        cmd+=['--sers-sanity-transform',mode]
        env=os.environ.copy();env.setdefault('OMP_NUM_THREADS','4')
        subprocess.run(cmd,cwd=ROOT,env=env,check=True)
