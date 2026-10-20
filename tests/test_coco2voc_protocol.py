from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from coco2voc_protocol import (CHANNEL_CATEGORY_IDS,VOC_TO_CATEGORY,VOC_TO_CHANNEL,
    STUFF_TO_CHANNEL,remap_voc_tags,add_histogram,protocol_scores)


def test_semantic_mapping_and_void_are_consistent_across_datasets():
    assert set(VOC_TO_CHANNEL[1:21].tolist())==set(range(61,81))
    for voc_id,category in enumerate(VOC_TO_CATEGORY):
        channel=int(VOC_TO_CHANNEL[voc_id])
        assert CHANNEL_CATEGORY_IDS[channel]==category
        if voc_id:assert STUFF_TO_CHANNEL[category-1]==channel
    np.testing.assert_array_equal(VOC_TO_CHANNEL[[0,1,9,15,255]],[0,65,76,61,255])
    np.testing.assert_array_equal(STUFF_TO_CHANNEL[[0,7,9,91,181,255]],[61,1,2,0,0,255])
    identity_tags=remap_voc_tags(np.eye(20,dtype=np.uint8))
    np.testing.assert_array_equal(identity_tags.sum(1),np.ones(20))
    np.testing.assert_array_equal(identity_tags.argmax(1)+1,VOC_TO_CHANNEL[1:21])
    assert not identity_tags[:,:60].any()


def test_metric_groups_and_old_predictions_on_voc_are_not_folded():
    histogram=np.zeros((81,81),np.int64)
    truth=np.array([[65,65,0,0,1,1,255]])
    prediction=np.array([[1,65,1,0,1,1,80]])
    add_histogram(histogram,truth,prediction)
    result=protocol_scores(histogram)
    assert result['valid_pixels']==6
    assert result['old_60_miou']==50
    assert result['new_20_miou']==50
    assert result['all_miou']==50
    assert result['background_iou']==50
    voc=np.zeros((81,81),np.int64)
    add_histogram(voc,np.array([[65,65]]),np.array([[1,65]]))
    assert protocol_scores(voc)['new_20_miou']==50
    assert voc[65,1]==1
