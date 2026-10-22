"""One-channel feasible-direction reparameterization, before existing arm target.

No PD/limit/reference/gain edits. The actor's a is tanh-normalized. Save mapping
age separately from existing permission age; alpha0 preserves original input.
"""
import torch

def remap_left_roll(action22, mapping_age_s, postcross_role, duration_s=.3):
    if action22.ndim!=2 or action22.shape[1]!=22:
        raise ValueError('expected [N,22] normalized actor action')
    if duration_s<=0:
        raise ValueError('positive blend duration required')
    age=torch.as_tensor(mapping_age_s,device=action22.device,dtype=action22.dtype)
    role=torch.as_tensor(postcross_role,device=action22.device,dtype=torch.bool)
    if age.shape!=(len(action22),) or role.shape!=age.shape:
        raise ValueError('age/role must have shape [N]')
    if not bool(torch.isfinite(action22).all() and torch.isfinite(age).all()):
        raise FloatingPointError('nonfinite action or age')
    if bool((action22.abs()>1.000001).any()):
        raise ValueError('requires tanh normalized actions, not unbounded Gaussian samples')
    t=(age/duration_s).clamp(0,1)
    alpha=t*t*(3-2*t)*role.to(action22.dtype)
    old=action22[:,15]
    inward=(old+1)*.5
    mapped=old+alpha*(inward-old)
    out=action22.clone()
    # Preserve original representation exactly at birth/replay, including -0.
    out[:,15]=torch.where(alpha==0,old,mapped)
    return out,dict(blend_alpha=alpha,left_roll_actor_action=old,
                    left_roll_mapped_action=out[:,15])

def advance_mapping_ticks(ticks, was_eligible, eligible):
    """Call once after physics/features, never before the command for that tick.

    Eligibility must be postrole & actual-cross latch & sourcephase>=11.02.
    A newly eligible state starts at0; only previously eligible states age.
    Reset must clear ticks/latch; sourceborn-eligible initializes age0/latchTrue.
    """
    if ticks.shape!=was_eligible.shape or ticks.shape!=eligible.shape:
        raise ValueError('mapping state shapes differ')
    if ticks.dtype!=torch.long or was_eligible.dtype!=torch.bool or eligible.dtype!=torch.bool:
        raise TypeError('ticks int64 and eligibility bool required')
    return torch.where(eligible & was_eligible,ticks+1,torch.zeros_like(ticks)),eligible.clone()
