"""Verify accumulated PBA/PTC gradients equal one effective batch of eight."""
import importlib.util
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.background_ambiguity_soft import continuous_background_loss

spec = importlib.util.spec_from_file_location('accumulation', ROOT / 'tools/pba_suite_accumulation.py')
accumulation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accumulation)


def test_accumulated_gradients_and_values_match_effective_batch():
    torch.manual_seed(13)
    inputs = torch.randn(8, 5, 3, 4, dtype=torch.float64)
    labels = torch.randint(0, 4, (8, 3, 4))
    labels[0, :2] = 255
    labels[2] = 255  # Unequal denominator, including an all-ignore image.
    labels[6, :, 1:] = 255
    valid = labels != 255
    cams = torch.rand(8, 3, 2, 2)
    auxiliary = torch.rand_like(cams)
    tags = torch.ones(8, 3)
    affinity = torch.randint(-1, 2, (8, 12, 12))
    affinity[0] = 1
    affinity[2] = -1
    positives = (affinity == 1).sum()
    negatives = (affinity == 0).sum()
    initial = torch.randn(4, 5, dtype=torch.float64)
    answers = []
    for world in (1, 2, 4):
        weight = initial.clone().requires_grad_()
        total = 0
        for rank in range(world):
            local_batch = 8 // world
            for offset in range(rank * local_batch, (rank + 1) * local_batch, 2):
                sl = slice(offset, offset + 2)
                logits = torch.einsum('oc,bchw->bohw', weight, inputs[sl])
                pba, _ = continuous_background_loss(logits, labels[sl], cams[sl], auxiliary[sl],
                                                   tags[sl], valid[sl], old_classes=1)
                pixels = accumulation.pixel_contribution(pba, valid[sl].sum(), valid.sum(), world)
                ptc = accumulation.ptc_contribution(logits, affinity[sl], positives, negatives,
                                                    world, 2 / local_batch)
                cls = F.multilabel_soft_margin_loss(logits.mean((2, 3))[:, 1:], tags[sl]) * (2 / local_batch)
                total = total + (pixels * .2 + ptc * .2 + cls) / world
        total.backward()
        answers.append((total.detach(), weight.grad.clone()))
    weight = initial.clone().requires_grad_()
    logits = torch.einsum('oc,bchw->bohw', weight, inputs)
    pba, _ = continuous_background_loss(logits, labels, cams, auxiliary, tags, valid, old_classes=1)
    features = F.normalize(logits.flatten(2), dim=1, eps=1e-8)
    cosine = (features.transpose(1, 2) @ features).abs()
    ptc = .5 * (1 - ((affinity == 1) * cosine).sum() / (positives + 1))
    ptc += .5 * ((affinity == 0) * cosine).sum() / (negatives + 1)
    cls = F.multilabel_soft_margin_loss(logits.mean((2, 3))[:, 1:], tags)
    whole = .2 * pba + .2 * ptc + cls
    whole.backward()
    for loss, gradient in answers:
        torch.testing.assert_close(loss, whole, atol=1e-8, rtol=1e-7)
        torch.testing.assert_close(gradient, weight.grad, atol=1e-8, rtol=1e-7)
