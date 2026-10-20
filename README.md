# EvoProto

Code for **EvoProto: Confusion-Aware Prototype Evolution for Weakly Supervised Incremental Segmentation**. The full method includes prototype segmentation, RCPL, directional CPA, and continuous PBA.

Use Python 3.10+ and a CUDA-compatible PyTorch/torchvision installation, then run `pip install -r requirements.txt`. Prepare VOC2012 with `JPEGImages`, `SegmentationClass`, and `SegmentationClassAug`. Split lists and image-level labels are included; images, checkpoints, logs, and experimental results are excluded. The ViT-B/16 ImageNet pretrained weights download automatically.

```bash
# Train all stages and evaluate the final checkpoint with TTA.
python train.py --task 10-5 --protocol overlap --data-root /path/to/VOC2012 --gpus 0
# Inspect the commands without starting training.
python train.py --task 10-5 --data-root /path/to/VOC2012 --dry-run
```

Tasks: `10-10`, `10-5`, `10-2`, `15-5`; protocols: `overlap`, `disjoint`. COCO→VOC uses `--task coco2voc --coco-image-root /path/to/coco --coco-mask-root /path/to/cocostuff` (2017 splits, zero-based COCO-Stuff masks). See `python train.py --help` and `python tools/evaluate.py --help`. Settings are in `configs/`; output goes to `runs/`.

Built on [ToCo](https://github.com/rulixiang/ToCo). Existing source notices are retained.
