"""Evaluate a released VOC or COCO→VOC checkpoint; no training logs are needed."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def iou_scores(histogram, names, initial_foreground):
    import numpy as np
    intersection = np.diag(histogram).astype(np.float64)
    union = histogram.sum(0) + histogram.sum(1) - intersection
    iou = np.divide(100 * intersection, union, out=np.full(len(names), np.nan), where=union > 0)
    def mean(values):
        valid = values[np.isfinite(values)]
        return float(valid.mean()) if len(valid) else None
    return dict(all_miou=mean(iou), old_miou=mean(iou[1:initial_foreground + 1]),
                new_miou=mean(iou[initial_foreground + 1:]),
                class_iou={name: float(v) if np.isfinite(v) else None for name, v in zip(names, iou)},
                valid_pixels=int(histogram.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=['tta', 'trainer', 'single'], default='tta')
    parser.add_argument('--tta-config', type=Path, default=ROOT / 'configs/tta.json')
    parser.add_argument('--data-root', type=Path, help='Override VOC data path after moving machines')
    parser.add_argument('--coco-image-root', type=Path)
    parser.add_argument('--coco-mask-root', type=Path)
    parser.add_argument('--limit', type=int, default=0, help='Positive value marks a partial smoke evaluation')
    parser.add_argument('--save-predictions', action='store_true')
    cli = parser.parse_args()
    if cli.limit < 0:
        parser.error('--limit must be nonnegative')
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.runtime/mpl'))
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    import tasks
    from datasets import voc
    from model.model_seg_neg import network
    from utils.camutils import multi_scale_seg
    from utils.tta import sliding_window_tta
    from coco2voc_protocol import CocoImages, VocEvaluation, add_histogram, protocol_scores

    config = json.loads(cli.stage_config.read_text())
    if cli.data_root:
        config.update(data_folder=str(cli.data_root.resolve()),
                      val_label_dir=str(cli.data_root.resolve() / 'SegmentationClass'))
    coco = config['dataset'] == 'coco2voc'
    config['list_folder'] = str(ROOT / ('datasets/coco2voc/voc' if coco else
        'datasets/splits/disjoint/voc' if config.get('protocol') == 'disjoint' else 'datasets/voc'))
    if coco:
        config['coco_protocol_manifest'] = str(ROOT / 'datasets/coco2voc/manifest.json')
        for name in ('coco_image_root', 'coco_mask_root'):
            if getattr(cli, name):
                config[name] = str(getattr(cli, name).resolve())
    args = argparse.Namespace(**config)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    blocks = tasks.get_per_task_classes(args.dataset, args.task, args.step)
    channels = sum(blocks)
    model = network(args.backbone, num_classes=channels, classes_list=blocks,
                    pretrained=False, init_momentum=args.momentum, aux_layer=args.aux_layer,
                    prototype_temperature=args.proto_temperature)
    checkpoint = torch.load(cli.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict({k.removeprefix('module.'): v for k, v in checkpoint['model_state'].items()}, strict=True)
    iteration = checkpoint.get('iteration')
    del checkpoint
    model.cuda().eval().requires_grad_(False)
    if coco:
        if args.step != 1:
            parser.error('Four-column COCO→VOC evaluation requires the final incremental stage')
        datasets = {'COCO': CocoImages(args, 'val2017'), 'VOC': VocEvaluation(args)}
    else:
        dataset = voc.VOC12SegDataset(root_dir=args.data_folder, name_list_dir=args.list_folder,
            tasks=args.task, step=args.step, split=args.val_set, stage='val', aug=False,
            ignore_index=args.ignore_index, num_classes=args.num_classes)
        dataset.label_dir = args.val_label_dir
        datasets = {'VOC': dataset}
    parameters = json.loads(cli.tta_config.read_text())['tta_parameters']
    output = cli.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    histograms = {}
    with torch.inference_mode():
        for name, dataset in datasets.items():
            if cli.limit:
                dataset.name_list = dataset.name_list[:cli.limit]
            histogram = np.zeros((channels, channels), np.int64)
            predictions = output / f'predictions_{name.lower()}'
            if cli.save_predictions:
                predictions.mkdir()
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
            for index, (names, images, labels, _) in enumerate(loader, 1):
                truth = labels[0].numpy()
                images = images.cuda()
                if cli.mode == 'tta':
                    scores = sliding_window_tta(model, images, channels, scales=parameters['scales'],
                        crop_size=parameters['crop_size'], stride=parameters['stride'], max_size=parameters['max_size'])
                else:
                    inputs = F.interpolate(images, (args.crop_size, args.crop_size), mode='bilinear', align_corners=False)
                    logits = multi_scale_seg(model, inputs, args.cam_scales) if cli.mode == 'trainer' and args.step else model(inputs)[1]
                    scores = F.interpolate(logits, truth.shape, mode='bilinear', align_corners=False)
                prediction = scores.argmax(1)[0].cpu().numpy().astype(np.uint8)
                add_histogram(histogram, truth, prediction)
                if cli.save_predictions:
                    Image.fromarray(prediction).save(predictions / (names[0] + '.png'))
                if index % 50 == 0 or index == len(dataset):
                    print(f'{name}: {index}/{len(dataset)}', flush=True)
            histograms[name] = histogram
    result = dict(task=args.task, step=args.step, iteration=iteration, mode=cli.mode,
                  checkpoint=str(cli.checkpoint.resolve()), partial_validation=bool(cli.limit),
                  image_tag_gating=False, dense_crf=False)
    if coco:
        a, b = protocol_scores(histograms['COCO']), protocol_scores(histograms['VOC'])
        result.update(coco_old60_miou=a['old_60_miou'], coco_new20_miou=a['new_20_miou'],
                      coco_all81_miou=a['all_miou'], voc_new20_miou=b['new_20_miou'], coco=a, voc=b)
    else:
        initial = tasks.get_per_task_classes(args.dataset, args.task, 0)[0] - 1
        result.update(iou_scores(histograms['VOC'], voc.class_list[:channels], initial))
    np.savez_compressed(output / 'confusion_counts.npz', **histograms)
    (output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
