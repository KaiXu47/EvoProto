"""Semantic channel mapping and dataset adapters for the frozen PBA method."""
import json
import logging
from pathlib import Path
import types

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

IGNORE_CATEGORY_IDS = {12,26,29,30,45,66,68,69,71,83,91}
NEW_CATEGORY_IDS = [1,2,3,4,5,6,7,9,16,17,18,19,20,21,44,62,63,64,67,72]
OLD_CATEGORY_IDS = [c for c in range(1,91) if c not in IGNORE_CATEGORY_IDS and c not in NEW_CATEGORY_IDS]
CHANNEL_CATEGORY_IDS = [0]+OLD_CATEGORY_IDS+NEW_CATEGORY_IDS
CATEGORY_TO_CHANNEL = {c:i for i,c in enumerate(CHANNEL_CATEGORY_IDS)}
# VOC ID -> original COCO category ID, from the official WILSON conversion.
VOC_TO_CATEGORY = [0,5,2,16,9,44,6,3,17,62,21,67,18,19,4,1,64,20,63,7,72]
VOC_TO_CHANNEL = np.full(256,255,np.uint8)
for voc_id,category_id in enumerate(VOC_TO_CATEGORY):
    VOC_TO_CHANNEL[voc_id] = CATEGORY_TO_CHANNEL[category_id]
STUFF_TO_CHANNEL = np.zeros(256,np.uint8)
for category_id,channel in CATEGORY_TO_CHANNEL.items():
    if category_id:STUFF_TO_CHANNEL[category_id-1] = channel
STUFF_TO_CHANNEL[255] = 255
assert len(OLD_CATEGORY_IDS)==60 and len(NEW_CATEGORY_IDS)==20
assert VOC_TO_CHANNEL[1]==65 and VOC_TO_CHANNEL[15]==61 and STUFF_TO_CHANNEL[0]==61


def read_mask(path):
    with Image.open(path) as image:
        mask=np.asarray(image).copy()
    if mask.ndim!=2:raise ValueError(f'Expected class-index PNG: {path}')
    return mask


def remap_voc_tags(tags):
    tags=np.asarray(tags)
    if tags.shape[-1]!=20:raise ValueError('Expected 20 foreground VOC image labels')
    result=np.zeros((*tags.shape[:-1],80),dtype=tags.dtype)
    result[...,VOC_TO_CHANNEL[1:21].astype(int)-1]=tags
    return result


def class_tags(mask):
    values=np.unique(mask);values=values[(values>0)&(values<81)]
    result=np.zeros(80,np.uint8);result[values.astype(int)-1]=1
    return result


def protocol_scores(histogram):
    """All includes background; foreground old/new exclude it. No head gating."""
    histogram=np.asarray(histogram,np.float64)
    intersection=np.diag(histogram)
    union=histogram.sum(0)+histogram.sum(1)-intersection
    # Follow WILSON's GT-support rule for missing validation classes.
    supported=histogram.sum(1)>0
    iou=np.divide(100*intersection,union,out=np.full_like(intersection,np.nan),where=union>0)
    def mean(start,end):
        selected=iou[start:end][supported[start:end]]
        return float(selected.mean()) if len(selected) else None
    return dict(old_60_miou=mean(1,min(61,len(iou))),
                new_20_miou=mean(61,len(iou)),all_miou=mean(0,len(iou)),
                background_iou=float(iou[0]) if supported[0] else None,
                iou=[float(v) if m else None for v,m in zip(iou,supported)],
                gt_supported_channels=np.flatnonzero(supported).tolist(),valid_pixels=int(histogram.sum()))


def add_histogram(histogram,truth,prediction):
    channels=len(histogram);valid=(truth>=0)&(truth<channels)
    assert truth.shape==prediction.shape
    assert np.all((prediction>=0)&(prediction<channels))
    histogram+=np.bincount(truth[valid].astype(np.int64)*channels+prediction[valid],
                           minlength=channels*channels).reshape(channels,channels)


class CocoImages(Dataset):
    def __init__(self,args,split,train=False):
        self.args,self.split,self.train=args,split,train
        meta=json.loads(Path(args.coco_protocol_manifest).read_text())
        self.name_list=meta['train_names'] if train else meta['validation_names']
        tag_path=Path(meta['train_tags'])
        if not tag_path.is_absolute():
            tag_path=Path(args.coco_protocol_manifest).resolve().parent/tag_path
        self.tags=np.load(tag_path,allow_pickle=True).item() if train else None
        self.label_dir=str(Path(args.coco_mask_root)/split)
        if train:
            from datasets import transforms
            self.color_jitter=transforms.PhotoMetricDistortion()
    def __len__(self):return len(self.name_list)
    def __getitem__(self,index):
        from datasets import transforms
        name=self.name_list[index]
        with Image.open(Path(self.args.coco_image_root)/self.split/(name+'.jpg')) as image:
            rgb=np.asarray(image.convert('RGB')).copy()
        mask=STUFF_TO_CHANNEL[read_mask(Path(self.args.coco_mask_root)/self.split/(name+'.png'))]
        assert rgb.shape[:2]==mask.shape
        if self.args.step==0:
            if self.train:assert not np.any((mask>=61)&(mask<=80)),f'Novel object in base training image: {name}'
            mask[(mask>=61)&(mask<=80)]=self.args.future_class_label
        tags=self.tags[name] if self.train else class_tags(mask)
        if self.train:
            rgb,mask=transforms.random_fliplr(rgb,mask)
            rgb=self.color_jitter(rgb)
            rgb,mask,_=transforms.random_crop(rgb,mask,crop_size=self.args.crop_size,
                     mean_rgb=[123.675,116.28,103.53],ignore_index=self.args.ignore_index)
        rgb=transforms.normalize_img(rgb).transpose(2,0,1)
        return name,rgb,mask,tags


class VocEvaluation(Dataset):
    def __init__(self,args):
        self.args=args
        meta=json.loads(Path(args.coco_protocol_manifest).read_text())
        self.name_list=meta['voc_validation_names']
    def __len__(self):return len(self.name_list)
    def __getitem__(self,index):
        from datasets import transforms
        name=self.name_list[index]
        with Image.open(Path(self.args.data_folder)/'JPEGImages'/(name+'.jpg')) as image:
            rgb=np.asarray(image.convert('RGB')).copy()
        gt=VOC_TO_CHANNEL[read_mask(Path(self.args.val_label_dir)/(name+'.png'))]
        return name,transforms.normalize_img(rgb).transpose(2,0,1),gt,class_tags(gt)


def make_datasets(args):
    import tasks
    from datasets import voc
    assert tasks.get_task_dict('coco2voc','coco2voc',1)=={0:[0]+OLD_CATEGORY_IDS,1:NEW_CATEGORY_IDS}
    if args.step==0:
        dataset=CocoImages(args,'train2017',train=True)
    else:
        # This existing class reads images + precomputed image labels only.
        # It never opens incremental pixel annotations.
        dataset=voc.VOC12ClsDataset(root_dir=args.data_folder,name_list_dir=args.list_folder,
            tasks=args.task,step=args.step,ignore_index=args.ignore_index,num_classes=81,
            split=args.train_set,stage='train',aug=True,crop_size=args.crop_size,
            rescale_range=args.scales,img_fliplr=True)
    return dataset,CocoImages(args,'val2017',train=False)


@torch.no_grad()
def validate(trainer,model=None,data_loader=None,args=None):
    from utils.camutils import multi_scale_seg
    channels=trainer.total_classes+1
    histogram=np.zeros((channels,channels),np.int64)
    model.eval()
    for _,inputs,labels,_ in data_loader:
        inputs=inputs.to(trainer.device)
        inputs=F.interpolate(inputs,(args.crop_size,args.crop_size),mode='bilinear',align_corners=False)
        logits=multi_scale_seg(model,inputs,args.cam_scales) if args.step else model(inputs)[1]
        truth=labels[0].numpy()
        prediction=F.interpolate(logits,truth.shape,mode='bilinear',align_corners=False).argmax(1)[0].cpu().numpy()
        add_histogram(histogram,truth,prediction)
    result=protocol_scores(histogram)
    result.update(step=args.step,iteration=trainer.current_iteration,
                  evaluation_dataset='COCO2017_full_validation',images=len(data_loader.dataset),
                  output_channels=channels,background_channel=0,void_label=255,
                  image_tag_gating=False,prototype_temperature=model.decoder.temperature)
    with (Path(args.work_dir)/'metrics.jsonl').open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n')
    np.savez_compressed(Path(args.work_dir)/f'validation_{trainer.current_iteration}.npz',histogram=histogram)
    logging.info('COCO validation: %s',json.dumps(result,allow_nan=False))
    model.train()
    return 0.0,json.dumps({key:result[key] for key in ['old_60_miou','new_20_miou','all_miou']})


def install_validation(trainer,args):
    trainer.validate=types.MethodType(validate,trainer)
