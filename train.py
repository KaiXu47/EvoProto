"""Launch the released full method without changing its training objectives."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STAGE_ITERATIONS = {
    '10-10': [20000, 12000], '10-5': [20000, 8000, 8000],
    '10-2': [20000] + [6000] * 5, '15-5': [20000, 8000],
    'coco2voc': [40000, 20000],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=STAGE_ITERATIONS, default='10-5')
    parser.add_argument('--protocol', choices=['overlap', 'disjoint'], default='overlap')
    parser.add_argument('--data-root', type=Path, required=True, help='VOC2012 root')
    parser.add_argument('--coco-image-root', type=Path, help='COCO train2017/val2017 parent')
    parser.add_argument('--coco-mask-root', type=Path, help='COCO-Stuff train2017/val2017 parent')
    parser.add_argument('--gpus', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/default.json')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--start-step', type=int, default=0)
    parser.add_argument('--end-step', type=int)
    parser.add_argument('--prev-checkpoint', type=Path, help='Required when starting at an incremental stage')
    parser.add_argument('--micro-batch', type=int, help='Default: full per-GPU batch, without accumulation')
    parser.add_argument('--dry-run', action='store_true', help='Print a plan without writing files or using GPUs')
    parser.add_argument('--smoke', action='store_true', help='Two updates and short validation per stage')
    parser.add_argument('--skip-final-tta', action='store_true')
    cli = parser.parse_args()
    gpus = cli.gpus.split(',')
    if len(gpus) not in (1, 2, 4) or any(not g.isdigit() for g in gpus) or len(set(gpus)) != len(gpus):
        parser.error('--gpus must name 1, 2, or 4 distinct GPU IDs')
    iterations = STAGE_ITERATIONS[cli.task]
    last = len(iterations) - 1 if cli.end_step is None else cli.end_step
    if not 0 <= cli.start_step <= last < len(iterations):
        parser.error('Invalid start/end stage')
    if cli.start_step and cli.prev_checkpoint is None:
        parser.error('--prev-checkpoint is required when --start-step > 0')
    coco = cli.task == 'coco2voc'
    if coco and (cli.coco_image_root is None or cli.coco_mask_root is None):
        parser.error('COCO→VOC requires both --coco-image-root and --coco-mask-root')
    if coco and cli.protocol != 'overlap':
        parser.error('COCO→VOC uses its own fixed protocol')
    base = json.loads(cli.config.read_text())
    data = cli.data_root.resolve()
    protocol = 'coco2voc' if coco else cli.protocol
    output = (cli.output_dir or ROOT / 'runs' / f'{protocol}_{cli.task}_{datetime.now():%Y%m%d_%H%M%S}').resolve()
    lists = (ROOT / 'datasets/coco2voc/voc' if coco else
             ROOT / 'datasets/splits/disjoint/voc' if cli.protocol == 'disjoint' else ROOT / 'datasets/voc')
    predecessor = cli.prev_checkpoint.resolve() if cli.prev_checkpoint else None
    if predecessor is not None and not cli.dry_run and not predecessor.is_file():
        parser.error(f'Missing predecessor checkpoint: {predecessor}')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=cli.gpus, PYTHONDONTWRITEBYTECODE='1',
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '4'))
    for step in range(cli.start_step, last + 1):
        work = output / f'step{step}'
        batch = 16 if coco and step == 0 else 8
        local_batch = batch // len(gpus)
        micro = cli.micro_batch or local_batch
        if micro < 1 or local_batch % micro:
            parser.error(f'Micro-batch must divide the per-GPU batch {local_batch}')
        config = dict(base, task=cli.task, protocol=protocol, dataset='coco2voc' if coco else 'voc',
                      step=step, pretrained=step == 0, num_classes=81 if coco else 21,
                      lr=6e-5 if step == 0 else 2e-5, max_iters=iterations[step],
                      data_folder=str(data), list_folder=str(lists),
                      seg_label_dir=str(data / 'SegmentationClassAug'),
                      val_label_dir=str(data / 'SegmentationClass'), work_dir=str(work),
                      prev_checkpoint=str(predecessor) if predecessor else None,
                      spg=local_batch, global_batch_size=batch)
        if coco:
            metadata = ROOT / 'datasets/coco2voc'
            config.update(coco_image_root=str(cli.coco_image_root.resolve()),
                          coco_mask_root=str(cli.coco_mask_root.resolve()),
                          coco_protocol_manifest=str(metadata / 'manifest.json'),
                          class_mapping_sha256=hashlib.sha256((metadata / 'class_mapping.csv').read_bytes()).hexdigest())
        config_file = output / 'configs' / f'step{step}.json'
        engine = ROOT / 'tools' / ('train_pba_coco2voc_stage.py' if coco else 'train_pba_suite_stage.py')
        command = [sys.executable, '-B', '-m', 'torch.distributed.run', '--standalone',
                   f'--nproc_per_node={len(gpus)}', str(engine), '--source', str(ROOT),
                   '--config', str(config_file), '--micro-batch', str(micro)]
        if cli.smoke:
            command.append('--smoke')
        print(f'Step {step}: {iterations[step]} updates; total batch {batch}; micro-batch {micro}', flush=True)
        print(shlex.join(command), flush=True)
        if not cli.dry_run:
            work.mkdir(parents=True, exist_ok=False)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            config_file.write_text(json.dumps(config, indent=2) + '\n')
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        predecessor = work / 'checkpoints/model_final.pth'
    if not cli.skip_final_tta and not cli.smoke and last == len(iterations) - 1:
        command = [sys.executable, '-B', str(ROOT / 'tools/evaluate.py'),
                   '--stage-config', str(output / f'step{last}/config.json'),
                   '--checkpoint', str(predecessor), '--output-dir', str(output / 'final_tta')]
        print(shlex.join(command), flush=True)
        if not cli.dry_run:
            subprocess.run(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=gpus[0]), check=True)


if __name__ == '__main__':
    main()
