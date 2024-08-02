from phalp.utils import get_pylogger

log = get_pylogger(__name__)

from phalp.trackers.PHALP import PHALP
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torch


class PHALPFaceID(PHALP):

    def __init__(self, cfg):
        super(PHALPFaceID, self).__init__(cfg)

        self.mtcnn = MTCNN(image_size=224, margin=40, select_largest=True)
        self.facenet = InceptionResnetV1(pretrained='vggface2', classify=False, num_classes=None).eval()

        # freeze all the face weights
        for param in self.mtcnn.parameters():
            param.requires_grad = False
        for param in self.facenet.parameters():
            param.requires_grad = False


    
    def run_additional_models(self, image_frame, pred_bbox, pred_masks, pred_scores, pred_classes, frame_name, t_, measurments, gt_tids, gt_annots):
        # crop image_frame using pred_bbox
        crops = []
        for box in pred_bbox:
            y_min, x_min, y_max, x_max = map(int, [box[0], box[1], box[2], box[3]])
            crop = image_frame[y_min:y_max, x_min:x_max]
            crops.append(crop)

        embeds = [] 
        for crop in crops:
            img = Image.fromarray(crop)
            mtcnn_out = self.mtcnn(img)
            if mtcnn_out is not None:
                img_embedding = self.facenet(mtcnn_out.unsqueeze(0))
            else:
                crop_tensor = transforms.ToTensor()(img).to(torch.float32)
                img_embedding = self.facenet(crop_tensor.unsqueeze(0))
            embeds.append(img_embedding) # shape 1x512
        
        return embeds 
