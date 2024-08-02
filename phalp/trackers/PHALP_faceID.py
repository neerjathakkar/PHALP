from phalp.utils import get_pylogger

log = get_pylogger(__name__)

from phalp.trackers.PHALP import PHALP

class PHALPFaceID(PHALP):

    def __init__(self, cfg):
        super(PHALPFaceID, self).__init__(cfg)

    
    def run_additional_models(self, image_frame, pred_bbox, pred_masks, pred_scores, pred_classes, frame_name, t_, measurments, gt_tids, gt_annots):
        print("running FaceID...")
        return list(range(len(pred_scores)))
