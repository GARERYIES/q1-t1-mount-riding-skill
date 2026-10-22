"""Phase-window high-level permission; unchanged lower-level PD and bounds."""
import torch

def gain_from_accepted(accepted):
    # The existing maximum50mrad permission envelope is retained. The physical
    # contract scale remains200mrad; g is explicit phase/course authority.
    return .25*min(max(int(accepted),0)/12.,1.)

def permission_mask(roles, branch, gain):
    mask=torch.ones((len(roles),22),device=roles.device,dtype=torch.bool)
    g=torch.as_tensor(gain,device=roles.device).expand(roles.shape)
    mask[:,14:]=(roles & (g>0))[:,None] if branch=='B' else False
    return mask

def fixed_baseline(reference,low,high,ankle_lower):
    baseline=reference.clamp(low,high).clone()
    baseline[:,10]=torch.maximum(baseline[:,10],torch.as_tensor(ankle_lower+.030,device=baseline.device,dtype=baseline.dtype))
    baseline[:,15]=baseline[:,15].clamp_min(low[15]-.001+.020)
    return baseline

def target(action,reference,scale,low,high,ankle_lower,roles,branch,gain):
    applied=action.clamp(-1,1).clone()
    applied[:,14:]*=(roles.float()*gain if branch=='B' else torch.zeros_like(roles,dtype=action.dtype))[:,None]
    raw=reference+scale*applied
    # The admitted fixed PD command is the arm residual origin. In particular,
    # do not hide19mrad of positive left-roll authority behind the old inset.
    baseline=fixed_baseline(reference,low,high,ankle_lower)
    raw[:,14:]=baseline[:,14:]+scale[14:]*applied[:,14:]
    before=raw.clamp(low,high);after=before.clone()
    after[:,10]=torch.maximum(after[:,10],torch.as_tensor(ankle_lower+.030,device=after.device,dtype=after.dtype))
    after[:,15]=after[:,15].clamp_min(low[15]-.001+.020)
    if bool((after>high).any()):raise ValueError('original inset exceeds upper')
    return applied,before,after
