"""Saved-checkpoint reconstruction/generalization audit, separate from inference."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from experiments.refine_v5.learned import load_model
from experiments.refine_v5.train_v3 import angles,aggregate

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run);d=np.load(run/'dataset/examples.npz');torch.set_num_threads(2);results=[]
    if np.any(d['split']=='test'):raise ValueError('P6 test must remain excluded')
    for checkpoint in sorted((run/'models').glob('*.pt')):
        model,ck=load_model(checkpoint)
        for split in ['development','validation']:
            for domain in ['real','clean','randomized']:
                ids=np.flatnonzero((d['split']==split)&(d['domain']==domain))
                with torch.no_grad():pred,lv=model(*[torch.tensor(d[key][ids].astype('float32')) for key in ['geometry','image','base']])
                angle=angles(pred.numpy(),d['target'][ids]);base=angles(d['base'][ids],d['target'][ids]);mask=d['mask'][ids]
                results.append(dict(model=checkpoint.stem,split=split,domain=domain,examples=len(ids),unique_poses=len(set(d['case_id'][ids])),direction_before_deg=aggregate(base[:,:8],mask[:,:8]),direction_after_deg=aggregate(angle[:,:8],mask[:,:8]),visible_limbs_within10=float(((angle[:,:8]<10)*mask[:,:8]).sum()/max(mask[:,:8].sum(),1)),fully_observed_limb_examples=int(np.all(mask[:,:8]>0,axis=1).sum()),all8_limbs_within10_examples=int((np.all(angle[:,:8]<10,axis=1)&np.all(mask[:,:8]>0,axis=1)).sum()),all8_limbs_within20_examples=int((np.all(angle[:,:8]<20,axis=1)&np.all(mask[:,:8]>0,axis=1)).sum()),uncertainty_calibrated_probability=False))
    (run/'frame-evaluation.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
