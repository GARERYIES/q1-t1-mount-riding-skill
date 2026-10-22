"""Pure controller permission, no privileged physical gate or phase modification."""
import torch
from .arm_authority import target,fixed_baseline,gain_from_accepted

def postcross_gain(roles,actual_cross_event_latch,phase,permission_ticks,accepted):
    x=(permission_ticks.float()/15.).clamp(0,1)
    return roles.float()*actual_cross_event_latch.float()*(phase>=11.02).float()*gain_from_accepted(accepted)*x*x*(3-2*x)

def apply_postcross_action(action,reference,scale,low,high,ankle_lower,roles,crossed,phase,permission_ticks,accepted):
    gain=postcross_gain(roles,crossed,phase,permission_ticks,accepted)
    return target(action,reference,scale,low,high,ankle_lower,roles,'B',gain)


def advance_permission_ticks(ticks,was_eligible,roles,crossed,phase):
    eligible=roles & crossed & (phase>=11.02)
    new_ticks=torch.where(eligible & was_eligible,ticks+1,torch.zeros_like(ticks))
    return new_ticks,eligible
