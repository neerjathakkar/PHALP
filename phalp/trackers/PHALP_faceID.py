from phalp.utils import get_pylogger

log = get_pylogger(__name__)

from phalp.trackers.PHALP import PHALP
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torch
import os
import pickle


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
        
        self.save_embeddings = True
        if self.save_embeddings:
            self.log_dir = os.path.join('/home/neerja/neerja_PHALP/facenet_embeddings', self.cfg.video.source.split("/")[-1])
            os.makedirs(self.log_dir, exist_ok=True)


    
    def run_additional_models(self, image_frame, pred_bbox, pred_masks, pred_scores, pred_classes, frame_name, t_, measurments, gt_tids, gt_annots):
        # crop image_frame using pred_bbox
        crops = []
        for box in pred_bbox:
            y_min, x_min, y_max, x_max = map(int, [box[0], box[1], box[2], box[3]])

            height, width, _ = image_frame.shape
            y_min = max(0, y_min)
            x_min = max(0, x_min)
            y_max = min(height, y_max)
            x_max = min(width, x_max)

            crop = image_frame[x_min:x_max, y_min:y_max]
            crops.append(crop)

        embeds = [] 
        confs = []
        for crop in crops:
            try:
                img = Image.fromarray(crop)
            except:
                # save image
                import cv2
                cv2.imwrite("error_image.jpg", image_frame)
                # import ipdb; ipdb.set_trace()
                img = Image.fromarray(image_frame)
            # import ipdb; ipdb.set_trace()
            mtcnn_out, conf = self.mtcnn(img, return_prob=True)
            print(conf)

            if conf is not None:
                confs.append(conf)
            else:
                confs.append(0)
            if mtcnn_out is not None:
                print("MTCNN worked")
                # import ipdb; ipdb.set_trace()
                img_embedding = self.facenet(mtcnn_out.unsqueeze(0))
            else:
                print("MTCNN did not work")
                crop_tensor = transforms.ToTensor()(img).to(torch.float32)
                img_embedding = self.facenet(crop_tensor.unsqueeze(0))
            embeds.append(img_embedding) # shape 1x512

            if self.save_embeddings:
                frame_name = frame_name.split('.')[0]
                frame_name = frame_name.split('/')[-1]
                with open(os.path.join(self.log_dir, frame_name + '_emb.pkl'), 'wb') as f:
                    pickle.dump(img_embedding.detach().cpu().numpy(), f)
                with open(os.path.join(self.log_dir, frame_name + '_img.pkl'), 'wb') as f:
                    if mtcnn_out is not None:
                        # undo normalization
                        # import ipdb; ipdb.set_trace()
                        mtcnn_out = mtcnn_out * 128 + 127.5
                        img = mtcnn_out.permute(1, 2, 0).detach().cpu().numpy()
                        pickle.dump(img, f)
                        # save image 
                        import cv2
                        cv2.imwrite(os.path.join(self.log_dir, frame_name + '.jpg'), img)
                    else:
                        pickle.dump(crop, f)
                        import cv2 
                        cv2.imwrite(os.path.join(self.log_dir, frame_name + '.jpg'), crop)
        
        # return embeds, confs 
        return embeds
