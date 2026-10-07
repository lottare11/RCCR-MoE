from pathlib import Path
import json,re
import numpy as np,pandas as pd
from scipy.stats import t as tdist
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'/'raw_rebuild_20260916'; REF=ROOT/'data'/'reference_kegg_20260916'
OUT=ROOT/'results'/'raw_rebuild_pathway_convergence_symbolfix_highconf_v008_20260916'
if OUT.exists(): raise RuntimeError(f'Refusing overwrite: {OUT}')
OUT.mkdir(parents=True)
RNG=np.random.default_rng(1002); B=5000
def bh(p):
 p=np.asarray(p,float); n=len(p); o=np.argsort(p); q=np.empty(n); v=p[o]*n/np.arange(1,n+1); v=np.minimum.accumulate(v[::-1])[::-1]; q[o]=np.minimum(v,1); return q
def normpid(p):
 p=str(p).strip().replace('path:',''); return 'hsa'+p[-5:] if p.startswith('map') and len(p)>=8 else p
def zcols(x):
 m=x.mean(0); s=x.std(0,ddof=1); return (x-m)/np.where(s>1e-8,s,1)
def tvec(S,y):
 A=S[y==1]; H=S[y==0]; n1=len(A); n0=len(H); m1=A.mean(0); m0=H.mean(0); v1=A.var(0,ddof=1); v0=H.var(0,ddof=1); se=np.sqrt(v1/n1+v0/n0); return (m1-m0)/np.where(se>1e-12,se,np.inf)
# references
official={}; aliases={}; gpaths={}; pnames={}
for line in (REF/'genes_full.txt').read_text().splitlines():
 p=line.split('\t');
 if len(p)<4: continue
 gid=p[0].split(':')[-1]; names=[x.strip() for x in p[3].split(';',1)[0].split(',') if x.strip()]
 if names: official[names[0]]=gid
 for a in names: aliases.setdefault(a,set()).add(gid)
alias=dict(official)
for a,ids in aliases.items():
 if a not in alias and len(ids)==1: alias[a]=next(iter(ids))
for line in (REF/'gene_path_full.txt').read_text().splitlines():
 g,p=line.split('\t')[:2]; gpaths.setdefault(g.split(':')[-1],set()).add(normpid(p))
for line in (REF/'path_names.txt').read_text().splitlines():
 p,n=line.split('\t',1); pnames[normpid(p)]=n.replace(' - Homo sapiens (human)','')
y=pd.read_csv(DATA/'canonical_subjects.csv').label.to_numpy(int)
TM=pd.read_csv(DATA/'transcript_feature_metadata.csv',low_memory=False); X=np.log1p(np.asarray(np.load(DATA/'transcript_fpkm_139x62354.npy',mmap_mode='r'),dtype=np.float32))
prev=(np.expm1(X)>1).mean(0)>=.20; idx=np.where(prev & TM.gene_name.astype(str).isin(alias).to_numpy())[0]; gg={}
for i in idx: gg.setdefault(str(TM.iloc[i].gene_name),[]).append(i)
syms=sorted(gg); GZ=zcols(np.column_stack([X[:,gg[s]].mean(1) for s in syms])); ent=[alias[s] for s in syms]; e2j={e:j for j,e in enumerate(ent)}; pg={}
for e in ent:
 for p in gpaths.get(e,set()):
  if p in pnames: pg.setdefault(p,set()).add(e)
MM=pd.read_csv(DATA/'metabolomics_feature_metadata.csv',low_memory=False); MX=np.log1p(np.clip(np.asarray(np.load(DATA/'metabolomics_full_139x2388.npy',mmap_mode='r'),dtype=np.float32),0,None))
lvl=pd.to_numeric(MM.Level,errors='coerce').fillna(99).to_numpy(); valid=MM.KEGG_ID.astype(str).str.match(r'^C\d{5}$').to_numpy()&(lvl<=2)&(MX.var(0)>1e-12); mg={}
for i in np.where(valid)[0]: mg.setdefault(str(MM.iloc[i].KEGG_ID),[]).append(i)
cpds=sorted(mg); CZ=zcols(np.column_stack([MX[:,mg[c]].mean(1) for c in cpds])); c2j={c:j for j,c in enumerate(cpds)}; pc={}
for c,ii in mg.items():
 ps=set()
 for i in ii:
  for tok in re.split(r'[;,|]',str(MM.iloc[i].get('KEGG_MapID',''))):
   tok=tok.strip()
   if re.match(r'^(map|hsa)\d{5}$',tok): ps.add(normpid(tok))
 for p in ps:
  if p in pnames: pc.setdefault(p,set()).add(c)
shared=[]; TS=[]; MS=[]; sizes=[]
for p in sorted(set(pg)&set(pc)):
 gj=[e2j[e] for e in pg[p] if e in e2j]; cj=[c2j[c] for c in pc[p] if c in c2j]
 if 10<=len(gj)<=500 and 3<=len(cj)<=100:
  shared.append(p); TS.append(GZ[:,gj].mean(1)); MS.append(CZ[:,cj].mean(1)); sizes.append((len(gj),len(cj)))
TS=np.column_stack(TS); MS=np.column_stack(MS); tt=tvec(TS,y); mt=tvec(MS,y); same=np.sign(tt)==np.sign(mt); obs=np.where(same,np.minimum(np.abs(tt),np.abs(mt)),0.0)
count=np.zeros(len(shared),int); maxvals=np.empty(B,float)
for b in range(B):
 yp=RNG.permutation(y); a=tvec(TS,yp); m=tvec(MS,yp); j=np.where(np.sign(a)==np.sign(m),np.minimum(np.abs(a),np.abs(m)),0.0); count += (j>=obs); maxvals[b]=j.max()
pemp=(count+1)/(B+1); q=bh(pemp); pfwer=np.array([(1+np.sum(maxvals>=x))/(B+1) for x in obs])
rows=[]
for i,p in enumerate(shared): rows.append([p,pnames[p],sizes[i][0],sizes[i][1],tt[i],mt[i],obs[i],same[i],pemp[i],q[i],pfwer[i]])
out=pd.DataFrame(rows,columns=['pathway_id','pathway_name','n_genes','n_compounds','t_transcript','t_metabolite','joint_min_abs_t','same_direction','perm_p','perm_q','maxT_fwer_p']).sort_values(['perm_q','perm_p','joint_min_abs_t'],ascending=[True,True,False]); out.to_csv(OUT/'joint_pathway_permutation.csv',index=False)
summary={'n_shared_paths':len(shared),'permutations':B,'seed':1002,'joint_statistic':'min(abs(t_transcript),abs(t_metabolite)) if directions agree, else 0','perm_q05':int((out.perm_q<.05).sum()),'maxT_fwer05':int((out.maxT_fwer_p<.05).sum()),'top':out.head(10).to_dict('records'),'gene_mapping':'official KEGG symbol first; unique non-official alias fallback only','metabolite_filter':'KEGG Level 1/2 only','guardrail':'Same labels permuted jointly across layers; maxT controls family-wise search over shared pathways. Post-hoc biological validation, not causal evidence.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
