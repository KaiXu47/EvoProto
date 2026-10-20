# EvoProto

By Fang Kai.

Supporting aggregate results for **EvoProto: Confusion-Aware Prototype Evolution for Weakly Supervised Incremental Segmentation**.

## Results package

Download [EvoProto-results.zip](EvoProto-results.zip) and extract it. The package includes:

- VOC and COCO-to-VOC result summaries and VOC class-wise and stage-wise IoU.
- Component and shared-weight controls, parameter sweeps, and prototype/confusion diagnostics.
- Old-to-new prediction-error rates and resource measurements.
- Selected training settings, data definitions, provenance, and a validation script.

Run `python validate_data.py` from the extracted folder using Python 3.8 or later. It uses the standard library.

IoU and mIoU are percentages; their differences are percentage points. The included README and data dictionary define the class groups, protocols, precision, and coverage of each file. The package contains aggregate experimental results rather than the original VOC/COCO images, per-image predictions, training code, or model checkpoints.

## Citation and reuse

Please cite the EvoProto paper and identify the exact repository revision used. Publication metadata will be added when available. A reuse license has not yet been selected.
