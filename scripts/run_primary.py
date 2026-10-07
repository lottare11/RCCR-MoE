"""Run the manuscript's CCRM configuration without changing research code."""
from pathlib import Path
import argparse,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def command(config,data,out):
    cmd=[sys.executable,str(ROOT/'src/train_raw_rebuild_sersnorm_20260916.py'),
         '--raw-data-dir',str(Path(data).resolve()),'--out-dir',str(Path(out).resolve())]
    for key,value in config.items():
        flag='--'+key.replace('_','-')
        if isinstance(value,bool):
            if value:cmd.append(flag)
        else:cmd.extend([flag,str(value)])
    return cmd
def run(config,data,out):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise SystemExit(f'Refusing nonempty output directory: {out}')
    env=os.environ.copy();env.setdefault('OMP_NUM_THREADS','4');env.setdefault('MKL_NUM_THREADS','4')
    subprocess.run(command(config,data,out),cwd=ROOT,env=env,check=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/raw_rebuild_20260916')
    p.add_argument('--out-dir',type=Path,default=ROOT/'results/raw_rebuild_snv_ladder_direct_20260916')
    p.add_argument('--config',type=Path,default=ROOT/'configs/primary.json')
    a=p.parse_args();run(json.loads(a.config.read_text()),a.data_dir,a.out_dir)
