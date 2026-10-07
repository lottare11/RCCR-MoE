"""Run pre-specified normalization, cumulative-structure or modality variants."""
from pathlib import Path
import argparse,json
from run_primary import ROOT,run
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('group',choices=['normalization','structure','subsets'])
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/raw_rebuild_20260916')
    a=p.parse_args();base=json.loads((ROOT/'configs/primary.json').read_text())
    jobs=[]
    if a.group=='normalization':jobs=[(f'normalization/{v}',dict(base,sers_sample_norm=v)) for v in ['none','snv','l2']]
    elif a.group=='structure':
        for norm in ['snv','l2']:
            cfg=dict(base,sers_sample_norm=norm)
            for i,flag in enumerate([None,'no_moe_routing','no_consensus_token','no_refinement_head']):
                if flag:cfg[flag]=True
                jobs.append((f'structure/{norm}_S{i}',cfg.copy()))
    else:
        for norm in ['snv','l2']:
            for mask in ['100','010','001','110','101','011','111']:
                jobs.append((f'subsets/{norm}_{mask}',dict(base,sers_sample_norm=norm,train_mask=mask)))
    for name,cfg in jobs:run(cfg,a.data_dir,ROOT/'results'/name)
