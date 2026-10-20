"""Global ratios whose gradients account for DDP's averaging across ranks."""
import torch
import torch.distributed as dist


def distributed_ratio(numerator, denominator, denominator_bias=0):
    count = torch.as_tensor(denominator, dtype=numerator.dtype, device=numerator.device)
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return numerator / (count + denominator_bias).clamp_min(1)
    totals = torch.stack((numerator.detach(), count.detach())).clone()
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    divisor = (totals[1] + denominator_bias).clamp_min(1)
    local_gradient = dist.get_world_size() * numerator / divisor
    return local_gradient + (totals[0] / divisor - local_gradient.detach())
