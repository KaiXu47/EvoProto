"""COCO→VOC adapter: same frozen PBA losses/optimizer, semantic dataset mapping."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta
import json
import logging
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from pba_suite_accumulation import pixel_contribution, ptc_contribution


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validation_control_group(timeout_seconds=7200):
    """Keep long rank-zero validation waits off the CUDA process group."""
    return dist.new_group(backend='gloo', timeout=timedelta(seconds=timeout_seconds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--micro-batch', type=int, default=2)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--audit-active-loss', action='store_true')
    cli = parser.parse_args()
    source = cli.source.resolve()
    os.environ['TMPDIR'] = os.environ.get('EVOPROTO_TMPDIR', f'/tmp/eps_{os.getppid()}')
    Path(os.environ['TMPDIR']).mkdir(parents=True, exist_ok=True)
    for variable, folder in [('MPLCONFIGDIR', 'mpl'), ('CUDA_CACHE_PATH', 'cuda')]:
        cache = source / '.runtime' / folder
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(cache)
    sys.path.insert(0, str(source))
    from continual.Trainer import Trainer
    from continual.prototype_supervision import build_incremental_supervision
    from datasets import voc
    from model.PAR import PAR
    from model.background_ambiguity_soft import continuous_background_loss
    from model.losses import cpa_loss, rcpl_loss
    from utils.camutils import cam_to_label, label_to_aff_mask
    from utils.prototype_confusion import confusion_update_due
    from utils.pyutils import setup_logger

    config = json.loads(cli.config.read_text())
    if config.get('pipeline_smoke', False):
        cli.smoke = True
        cli.audit_active_loss = True
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    global_batch = config.get('global_batch_size', 8)
    assert isinstance(global_batch, int) and global_batch > 0
    assert world in (1, 2, 4) and global_batch % world == 0
    local_batch = global_batch // world
    assert local_batch % cli.micro_batch == 0
    config.update(local_rank=local_rank, spg=local_batch, w_proto_seg=config['w_seg'],
                  micro_batch_size=cli.micro_batch, gradient_accumulation=local_batch // cli.micro_batch,
                  gradient_accumulation_enabled=local_batch != cli.micro_batch,
                  global_batch_size=global_batch, world_size=world,
                  distributed_timeout_seconds=7200, validation_barrier_backend='gloo',
                  iteration_unit='optimizer_updates',
                  engine='direct_batch' if local_batch == cli.micro_batch else 'exact_effective_batch_accumulation')
    if cli.smoke:
        config.update(max_iters=2, eval_iters=2, log_iters=1, warmup_iters=0,
                      loss_warmup_iters=0, confusion_start_iter=0 if cli.audit_active_loss else 4000,
                      train_limit=max(16, 2 * global_batch), val_limit=2)
    args = argparse.Namespace(**config)
    args.ckpt_dir = str(Path(args.work_dir) / 'checkpoints')
    torch.cuda.set_device(local_rank)
    if cli.smoke:
        torch.cuda.reset_peak_memory_stats()
    timeout = timedelta(seconds=args.distributed_timeout_seconds)
    dist.init_process_group('nccl', timeout=timeout)
    # Whole-set validation can exceed NCCL's default ten-minute watchdog.
    # Waiting ranks must stay on a CPU group with no pending CUDA collective.
    control_group = validation_control_group(args.distributed_timeout_seconds)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.hub.set_dir(str(source / 'pretrained'))
    if rank == 0:
        Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
        setup_logger(filename=str(Path(args.work_dir) / 'train.log'))
        write_json(Path(args.work_dir) / 'config.json', vars(args))
        logging.info('Effective batch=%d; GPUs=%d; microbatch=%d; accumulation=%d; iterations count optimizer updates',
                     global_batch, world, cli.micro_batch, args.gradient_accumulation)
        logging.info('Frozen configuration: %s', json.dumps(vars(args)))
    dist.barrier()
    trainer = Trainer(args)
    from coco2voc_protocol import make_datasets, install_validation
    install_validation(trainer,args)
    if args.step > 0:
        trainer.load_step_ckpt(args.prev_checkpoint)
        trainer.model_old.to(trainer.device).eval()
    trainer.model.to(trainer.device)
    model = DDP(trainer.model, device_ids=[local_rank], find_unused_parameters=True)
    optim = trainer.optimizer
    par = PAR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).to(trainer.device) if args.step else None
    dataset, validation = make_datasets(args)
    if args.train_limit:
        dataset.name_list = dataset.name_list[:args.train_limit]
    if args.val_limit:
        validation.name_list = validation.name_list[:args.val_limit]
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    generator = torch.Generator().manual_seed(args.seed + rank)
    loader = DataLoader(dataset, batch_size=local_batch, sampler=sampler, drop_last=True,
                        num_workers=args.num_workers, prefetch_factor=2, generator=generator)
    val_loader = DataLoader(validation, batch_size=1, shuffle=False, num_workers=2)
    # Disjoint rank shards avoid duplicated padding images in the statistics.
    cal_loader = DataLoader(dataset, batch_size=cli.micro_batch,
                            sampler=list(range(rank, len(dataset), world)),
                            num_workers=2, drop_last=False)
    epoch = 0
    sampler.set_epoch(epoch)
    iterator = iter(loader)
    start = time.monotonic()
    interval = np.zeros(7, np.float64)
    interval_updates = 0
    first_gradient_audit = None
    for update in range(args.max_iters):
        trainer.current_iteration = update + 1
        if args.step and args.confusion_reweight and confusion_update_due(
                update, args.confusion_start_iter, args.confusion_interval):
            # Bare module has no rank-dependent forward collectives, allowing
            # unequal last confusion shard lengths without a DDP deadlock.
            trainer.confusion_matrix = trainer.cal_sim(model=model.module, data_loader=cal_loader, args=args)
            trainer.confusion_counterparts, trainer.confusion_weights = trainer.get_weight(
                trainer.confusion_matrix, trainer.total_classes, trainer.new_classes)
            if rank == 0:
                np.savez(Path(args.work_dir) / f'confusion_{update}.npz',
                         counts=trainer.confusion_counts.cpu().numpy(),
                         matrix=trainer.confusion_matrix.cpu().numpy(),
                         weights=trainer.confusion_weights.cpu().numpy())
                logging.info('Confusion refreshed after %d optimizer updates; counterparts=%s',
                             update, trainer.confusion_counterparts.tolist())
        try:
            data = next(iterator)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            iterator = iter(loader)
            data = next(iterator)
        prepared = []
        denominators = torch.zeros(3, dtype=torch.float64, device=trainer.device)
        # Targets and pair masks are prepared without graphs before gradient
        # accumulation so their global denominators span the full effective batch.
        with torch.no_grad():
            for offset in range(0, local_batch, cli.micro_batch):
                end = offset + cli.micro_batch
                inputs = data[1][offset:end].to(trainer.device)
                if args.step == 0:
                    labels = data[2][offset:end].to(trainer.device).long()
                    labels[(labels > trainer.total_classes) & (labels != args.ignore_index)] = args.future_class_label
                    tags = data[3][offset:end, :trainer.total_classes].to(trainer.device)
                    item = dict(labels=labels, tags=tags)
                else:
                    tags = data[2][offset:end, :trainer.total_classes].to(trainer.device)
                    box = data[3][offset:end]
                    sup = build_incremental_supervision(model.module, trainer.model_old, inputs,
                                                        tags, box, par, args,
                                                        trainer.total_classes, trainer.new_classes)
                    aux = F.interpolate(sup.cams_aux, (args.crop_size // 16,) * 2,
                                        mode='bilinear', align_corners=False)
                    _, pseudo = cam_to_label(aux, cls_label=sup.cls_labels, img_box=box,
                                             ignore_mid=True, bkg_thre=args.bkg_thre,
                                             high_thre=args.high_thre, low_thre=args.low_thre,
                                             ignore_index=args.ignore_index)
                    mask = label_to_aff_mask(pseudo)
                    item = dict(labels=sup.merged_labels, tags=sup.cls_labels,
                                cams=sup.cams, cams_aux=sup.cams_aux, valid=sup.valid_mask,
                                old_prototypes=sup.old_prototypes, affinity=mask)
                    denominators[1] += (mask == 1).sum()
                    denominators[2] += (mask == 0).sum()
                    del sup, aux, pseudo, mask
                denominators[0] += (item['labels'] != args.ignore_index).sum()
                prepared.append({key: value.cpu() for key, value in item.items()})
                del item, inputs, tags
        dist.all_reduce(denominators)
        optim.zero_grad(set_to_none=True)
        logged = torch.zeros(7, device=trainer.device)
        pba_counts = torch.zeros(7, dtype=torch.float64, device=trainer.device)
        for index, cached in enumerate(prepared):
            offset = index * cli.micro_batch
            inputs = data[1][offset:offset + cli.micro_batch].to(trainer.device)
            item = {key: value.to(trainer.device) for key, value in cached.items()}
            fraction = cli.micro_batch / local_batch
            context = model.no_sync() if index + 1 < len(prepared) else nullcontext()
            with context:
                if args.step == 0:
                    _, cls, segs, fmap, cls_aux, _, prototypes = model(inputs, cam_grad=True, n_iter=update)
                else:
                    cls, segs, fmap, cls_aux, _, prototypes = model(inputs, crops=[], n_iter=update)
                cls_loss = F.multilabel_soft_margin_loss(cls, item['tags']) * fraction
                auxiliary = F.multilabel_soft_margin_loss(cls_aux, item['tags']) * fraction
                separation = rcpl_loss(prototypes, trainer.confusion_weights,
                                       mode=args.proto_sep_mode, margin=args.proto_margin) * fraction
                segs = F.interpolate(segs, item['labels'].shape[-2:], mode='bilinear', align_corners=False)
                valid_count = (item['labels'] != args.ignore_index).sum()
                if args.step and args.soft_pba and update >= args.loss_warmup_iters:
                    seg_mean, stats = continuous_background_loss(
                        segs, item['labels'], item['cams'], item['cams_aux'], item['tags'], item['valid'],
                        trainer.old_classes, class_weight=trainer.class_weight,
                        ignore_index=args.ignore_index, strength=args.soft_pba_strength)
                    bg = stats['background_pixels']
                    pba_counts += pba_counts.new_tensor([stats['valid_pixels'], bg,
                        stats['mean_cam_ambiguity'] * bg, stats['mean_background_ambiguity'] * bg,
                        stats['mean_directional_weight'] * bg, stats['ambiguity_mass'], 1])
                else:
                    seg_mean = F.cross_entropy(segs, item['labels'], weight=trainer.class_weight,
                                               ignore_index=args.ignore_index, reduction='sum') / valid_count.clamp_min(1)
                segmentation = pixel_contribution(seg_mean, valid_count, denominators[0], world)
                preservation = segmentation * 0
                ptc = segmentation * 0
                if args.step:
                    preservation = cpa_loss(prototypes, item['old_prototypes'], trainer.confusion_weights,
                                            mode=args.proto_kd_mode, skip_background=args.cpa_skip_background) * fraction
                    ptc = ptc_contribution(fmap, item['affinity'], denominators[1], denominators[2], world, fraction)
                loss = cls_loss + auxiliary + args.w_ptc * ptc
                active = args.step == 0 or update >= args.loss_warmup_iters
                loss += (1 if active else 0) * (args.w_seg * segmentation + args.w_proto_kd * preservation
                                               + args.w_proto_sep * separation)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss step{args.step} update{update + 1}')
                loss.backward()
                logged += torch.stack([cls_loss, auxiliary, ptc, segmentation, preservation,
                                       separation, loss]).detach().float()
            del inputs, item, segs, fmap, cls, cls_aux, prototypes
        if cli.smoke and update == 0:
            gradients = [p.grad for p in trainer.model.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            first_gradient_audit = dict(tensors=len(gradients), all_finite=True,
                                       global_pixel_count=int(denominators[0]),
                                       global_positive_pairs=int(denominators[1]),
                                       global_negative_pairs=int(denominators[2]))
        optim.step()
        assert optim.global_step == update + 1
        dist.all_reduce(logged)
        logged /= world
        interval += logged.cpu().numpy()
        interval_updates += 1
        if (update + 1) % args.log_iters == 0 or update == args.loss_warmup_iters:
            dist.all_reduce(pba_counts)
            if rank == 0 and pba_counts[6] > 0:
                values = pba_counts.cpu().tolist()
                bg = max(values[1], 1)
                stats = dict(valid_pixels=int(values[0]), background_pixels=int(values[1]),
                             mean_cam_ambiguity=values[2] / bg, mean_background_ambiguity=values[3] / bg,
                             mean_directional_weight=values[4] / bg, ambiguity_mass=values[5],
                             pba_strength=args.soft_pba_strength, thresholds_used=False, prototypes_used=False,
                             step=args.step, iteration=update + 1, scope='global_effective_batch', global_batch_size=args.global_batch_size)
                with (Path(args.work_dir) / 'soft_pba.jsonl').open('a') as handle:
                    handle.write(json.dumps(stats) + '\n')
                logging.info('Continuous PBA: %s', stats)
        if (update + 1) % args.log_iters == 0:
            if rank == 0:
                values = interval / interval_updates
                logging.info('Iter: %d; elapsed: %.1fs; LR: %.3e; cls_loss: %.4f; cls_loss_aux: %.4f; '
                             'ptc_loss: %.4f; seg_loss: %.4f; proto_kd: %.6f; proto_sep: %.6f; total: %.4f',
                             update + 1, time.monotonic() - start, optim.param_groups[0]['lr'], *values)
                write_json(Path(args.work_dir) / 'progress.json', dict(iteration=update + 1, max_iters=args.max_iters,
                           elapsed_seconds=time.monotonic() - start, lr=optim.param_groups[0]['lr'],
                           global_batch_size=args.global_batch_size, world_size=world, accumulation=args.gradient_accumulation))
            interval.fill(0)
            interval_updates = 0
        if (update + 1) % args.eval_iters == 0 or update + 1 == args.max_iters:
            dist.barrier(group=control_group)
            if rank == 0:
                state = dict(model_state=trainer.model.state_dict(), iteration=update + 1,
                             global_batch_size=args.global_batch_size, iteration_unit='optimizer_updates')
                torch.save(state, Path(args.ckpt_dir) / f'model_iter_{update + 1}.pth')
                write_json(Path(args.work_dir) / 'progress.json', dict(
                    iteration=update + 1, max_iters=args.max_iters, phase='validating',
                    elapsed_seconds=time.monotonic() - start, global_batch_size=args.global_batch_size,
                    world_size=world, accumulation=args.gradient_accumulation))
                logging.info('Full validation begins after update %d; waiting ranks use Gloo with %ds timeout',
                             update + 1, args.distributed_timeout_seconds)
                score, table = trainer.validate(model=model.module, data_loader=val_loader, args=args)
                logging.info('val cls score: %.6f\n%s', score, table)
                write_json(Path(args.work_dir) / 'progress.json', dict(
                    iteration=update + 1, max_iters=args.max_iters, phase='training',
                    elapsed_seconds=time.monotonic() - start, global_batch_size=args.global_batch_size,
                    world_size=world, accumulation=args.gradient_accumulation))
            dist.barrier(group=control_group)
            model.train()
    if rank == 0:
        torch.save(dict(model_state=trainer.model.state_dict(), iteration=args.max_iters,
                        global_batch_size=args.global_batch_size, iteration_unit='optimizer_updates'),
                   Path(args.ckpt_dir) / 'model_final.pth')
        write_json(Path(args.work_dir) / 'completion.json', dict(status='completed',
                   iteration=args.max_iters, global_batch_size=args.global_batch_size, world_size=world,
                   accumulation=args.gradient_accumulation, smoke=cli.smoke,
                   gradient_audit=first_gradient_audit,
                   cuda_peak_memory=dict(allocated_bytes=torch.cuda.max_memory_allocated(),
                                         reserved_bytes=torch.cuda.max_memory_reserved()) if cli.smoke else None,
                   finished_at=datetime.now().astimezone().isoformat()))
        logging.info('Final checkpoint saved at optimizer update %d', args.max_iters)
    dist.barrier(group=control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
