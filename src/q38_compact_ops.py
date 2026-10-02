"""Compact transport helpers for the pinned SM80 vocabulary and finite top-k.
Keeps full-width softmax/cumsum to preserve reference reduction layout.
Called only inside the guarded target-sampling path.
"""
import torch
CAP=128

def pack(x, offset=0):
    v, i=x.topk(CAP+1,dim=-1,sorted=True)
    # FP32 exactly represents vocabulary indices in this model (<2**24).
    return torch.cat((v[:,:CAP].float(), (i[:,:CAP]+offset).float(),v[:,CAP:].float()),dim=-1)

def compact_filter(packs,vocab,k,p):
    vals=torch.cat([t[:,:CAP] for t in packs],-1)
    ids=torch.cat([t[:,CAP:2*CAP].long() for t in packs],-1)
    omitted=torch.stack([t[:,-1] for t in packs],-1).max(-1).values
    # Large CUDA full-vocabulary sort preserves input index order for equal
    # values in the pinned reference build. Explicitly reproduce that order.
    order=ids.argsort(dim=-1,stable=True)
    ids=ids.gather(1,order);vals=vals.gather(1,order)
    vals,order=vals.sort(dim=-1,stable=True);ids=ids.gather(1,order)
    kth=vals.gather(1,(vals.shape[-1]-k.long()).unsqueeze(-1))
    unsafe=(omitted>=kth.squeeze(-1)) | ~torch.isfinite(kth.squeeze(-1)) | torch.isnan(vals).any(-1) | torch.isposinf(vals).any(-1)
    if bool(unsafe.any().item()):return None
    vals=vals.masked_fill(vals<kth,-float('inf'))
    width=vals.shape[-1]
    dense=torch.full((vals.shape[0],vocab),-float('inf'),device=vals.device)
    dense[:,-width:]=vals
    probs=dense.softmax(-1)
    torch.cumsum(probs,-1,out=probs)
    mask=probs[:,-width:] <= 1-p.unsqueeze(-1)
    mask[:,-1]=False
    vals=vals.masked_fill(mask,-float('inf'))
    output=torch.full_like(dense,-float('inf'))
    return output.scatter_(1,ids,vals)
