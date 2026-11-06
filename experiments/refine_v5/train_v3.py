"""Train actual ablations; choose epochs on exploratory validation, not P6 test."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from experiments.refine_v5.learned import ResidualFrames,VERSION
from experiments.refine_v5.learning_data import BONES

def angles(a,b):return np.degrees(np.arccos(np.clip((a[:,:,1]*b[:,:,1]).sum(-1),-1,1)))
def aggregate(a,mask):
    return float((a*mask).sum()/max(mask.sum(),1))
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--epochs',type=int,default=140);a=p.parse_args();root=Path(a.run)
    d=np.load(root/'dataset/examples.npz');out=root/'models';out.mkdir(exist_ok=False);torch.set_num_threads(2)
    real=d['domain']=='real';dev=d['split']=='development';val=d['split']=='validation'
    if np.any(d['split']=='test'):raise ValueError('P6 test leakage')
    ts={k:torch.tensor(d[k].astype('float32')) for k in ['geometry','image','base','target','mask','synthetic']}
    vi=np.where(real&val)[0];baseangle=angles(d['base'][vi],d['target'][vi]);mask=d['mask'][vi]
    results=[]
    configs=[('real_geometry',False,['real']),('real_image',True,['real']),('synthetic_clean',True,['clean']),('synthetic_randomized',True,['randomized']),('mixed_randomized',True,['real','randomized'])]
    for name,useimage,domains in configs:
        torch.manual_seed(711);rng=np.random.default_rng(711);start=time.monotonic()
        model=ResidualFrames(d['geometry'].shape[1],d['image'].shape[1],useimage);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        ix=np.where(dev&np.isin(d['domain'],domains))[0];realix=ix[real[ix]];synix=ix[~real[ix]]
        best=None;bestscore=float('inf');history=[]
        for epoch in range(a.epochs):
            model.train()
            # Equal real/synthetic contribution: augmentation count cannot swamp real sketches.
            if len(realix) and len(synix):order=np.r_[synix,rng.choice(realix,len(synix),replace=True)]
            else:order=np.tile(ix,max(1,256//max(1,len(ix))))
            rng.shuffle(order);losses=[]
            for begin in range(0,len(order),128):
                ids=order[begin:begin+128];pred,logv=model(ts['geometry'][ids],ts['image'][ids],ts['base'][ids]);target=ts['target'][ids];m=ts['mask'][ids]
                direction=((pred[:,:,1]-target[:,:,1])**2).sum(-1)
                roll=((pred[:,:,0]-target[:,:,0])**2).sum(-1)
                rollweight=.05+.45*ts['synthetic'][ids,None]
                error=direction+rollweight*roll
                loss=(((error*torch.exp(-logv)+logv)*m).sum()/m.sum().clamp(min=1)+.025*((pred-ts['base'][ids])**2).mean())
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step();losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():pv,lv=model(ts['geometry'][vi],ts['image'][vi],ts['base'][vi])
            av=angles(pv.numpy(),d['target'][vi]);score=aggregate(av[:,:8],mask[:,:8])
            if score<bestscore:
                bestscore=score;best={k:v.detach().clone() for k,v in model.state_dict().items()};bestepoch=epoch
            if epoch%20==0:history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),val_visible_limb_deg=score))
        model.load_state_dict(best);model.eval()
        with torch.no_grad():pv,lv=model(ts['geometry'][vi],ts['image'][vi],ts['base'][vi])
        av=angles(pv.numpy(),d['target'][vi]);floors=[]
        for j in range(13):
            vals=np.radians(av[:,j][mask[:,j]>0]);floors.append(float(np.sqrt(np.mean(vals**2))) if len(vals) else float(np.pi))
        checkpoint=dict(version=VERSION,weights=best,geometry_dim=d['geometry'].shape[1],image_dim=d['image'].shape[1],use_image=useimage,validation_error_floor_rad=floors,train_cases=sorted(set(d['case_id'][ix].tolist())),training_domains=domains,test_used=False,camera_status='real_unknown_source_frame_hypothesis',image_features='frozen_engineered_not_pretrained')
        torch.save(checkpoint,out/(name+'.pt'))
        # Save development+validation predictions independent of the label evaluation file.
        ri=np.where(real)[0]
        with torch.no_grad():pp,ll=model(ts['geometry'][ri],ts['image'][ri],ts['base'][ri])
        predictions=[dict(case_id=str(d['case_id'][i]),axes=pp[k].tolist(),uncertainty_rad=np.maximum(np.exp(ll[k].numpy()/2),floors).tolist()) for k,i in enumerate(ri)]
        (out/(name+'.predictions.json')).write_text(json.dumps(predictions))
        records=[]
        for k,i in enumerate(vi):records.append(dict(case_id=str(d['case_id'][i]),before=aggregate(baseangle[k,:8],mask[k,:8]),after=aggregate(av[k,:8],mask[k,:8]),observed=int(mask[k,:8].sum()),per_bone_before=baseangle[k].tolist(),per_bone_after=av[k].tolist()))
        metric=dict(name=name,training_examples=len(ix),training_cases=len(set(d['case_id'][ix])),validation_cases=len(vi),visible_limb_before=aggregate(baseangle[:,:8],mask[:,:8]),visible_limb_after=bestscore,best_epoch=bestepoch,elapsed_seconds=time.monotonic()-start,per_bone={b:dict(before=aggregate(baseangle[:,j],mask[:,j]),after=aggregate(av[:,j],mask[:,j]),observed=int(mask[:,j].sum())) for j,b in enumerate(BONES)},records=records,history=history,independent_blind=False)
        results.append(metric);(out/'metrics.json').write_text(json.dumps(results,indent=2));print('V5_TRAINED',name,metric['visible_limb_before'],bestscore,flush=True)
    selected=min(results,key=lambda r:r['visible_limb_after'])
    (out/'selection.json').write_text(json.dumps(dict(selected=selected['name'],criterion='exploratory_validation_visible_limb_angle',test_used=False,production_promoted=False,identity_baseline=selected['visible_limb_before'],selected_error=selected['visible_limb_after']),indent=2))
if __name__=='__main__':main()
