"""Residual 6D bone-frame predictor; no training labels or GT loader at inference."""
import numpy as np
import torch
from torch import nn
VERSION='v5-p2-p3-masked-residual-frames-v1'
class ResidualFrames(nn.Module):
    def __init__(self,geometry_dim,image_dim,use_image=True):
        super().__init__();self.use_image=use_image
        self.net=nn.Sequential(nn.Linear(geometry_dim+(image_dim if use_image else 0),128),nn.SiLU(),nn.Dropout(.15),nn.Linear(128,96),nn.SiLU(),nn.Linear(96,91))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,g,image,base):
        x=torch.cat([g,image],-1) if self.use_image else g
        out=self.net(x);raw=base+out[:,:78].reshape(-1,13,2,3)
        x=nn.functional.normalize(raw[:,:,0],dim=-1);y=raw[:,:,1]-(raw[:,:,1]*x).sum(-1,keepdim=True)*x
        y=nn.functional.normalize(y,dim=-1)
        return torch.stack([x,y],2),out[:,78:].clamp(-4,3)

def load_model(path):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck['version']!=VERSION:raise ValueError('incompatible checkpoint')
    model=ResidualFrames(ck['geometry_dim'],ck['image_dim'],ck['use_image']);model.load_state_dict(ck['weights']);model.eval()
    return model,ck

def infer(path,geometry,image,base):
    model,ck=load_model(path)
    tensors=[torch.tensor(np.asarray(x,dtype='float32')[None]) for x in (geometry,image,base)]
    with torch.no_grad():axes,logv=model(*tensors)
    return dict(axes=axes[0].numpy(),uncertainty=np.maximum(np.exp(logv[0].numpy()/2),np.array(ck['validation_error_floor_rad'])),version=VERSION)
