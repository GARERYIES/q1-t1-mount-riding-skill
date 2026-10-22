"""Pure proposed split authority; no model/PD/reference/observation changes."""
import torch
from .postcross_authority import postcross_gain
from .arm_authority import target
from .left_roll_mapping import remap_left_roll
PRELOSS_PHASE=8.74
COMPARE_EPS_S=1e-10 # Comparison only; never round/mutate source physical/reference time.

def left_eligible(roles,phase,mode,crossed):
    if mode=='A':return roles & crossed & (phase>=11.02)
    if mode!='B':raise ValueError('mode A/B required')
    return roles & (phase+COMPARE_EPS_S>=PRELOSS_PHASE)

def split_gains(roles,crossed,phase,right_ticks,left_ticks,mode):
    right=postcross_gain(roles,crossed,phase,right_ticks,16)
    if mode=='A':left=right
    else:
        eligible=left_eligible(roles,phase,mode,crossed);x=(left_ticks.float()/15.).clamp(0,1)
        left=eligible.float()*.25*x*x*(3-2*x)
    return torch.cat((left[:,None].expand(-1,4),right[:,None].expand(-1,4)),1)

def action_mask(gains8):
    out=torch.ones((len(gains8),22),device=gains8.device,dtype=torch.bool);out[:,14:]=gains8>0;return out

def split_target(action,reference,scale,low,high,ankle_lower,roles,gains8,left_mapping_ticks):
    mapped,diag=remap_left_roll(action,left_mapping_ticks.float()*.02,roles)
    scaled=mapped.clamp(-1,1).clone();scaled[:,14:]*=gains8
    # Reuse the old fixed PD residual origin, inset and clamps unchanged. This
    # second factor is exactly1 for active roles (0 for support), not a new gain.
    result=target(scaled,reference,scale,low,high,ankle_lower,roles,'B',torch.ones(len(roles),device=roles.device))
    return (*result,diag)

def advance_left_ticks(ticks,was_eligible,eligible):
    new=torch.where(eligible & was_eligible,ticks+1,torch.zeros_like(ticks))
    return new,eligible
