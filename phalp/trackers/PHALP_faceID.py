from phalp.utils import get_pylogger

log = get_pylogger(__name__)

from phalp.trackers.PHALP import PHALP
from facenet_pytorch import MTCNN, InceptionResnetV1
import mediapipe as mp
from mediapipe.tasks import python as pythonmp
from mediapipe.tasks.python import vision

import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torch
import os
import pickle
import cv2
import numpy as np


class PHALPFaceID(PHALP):

    def __init__(self, cfg):
        super(PHALPFaceID, self).__init__(cfg)

        # self.mtcnn = MTCNN(image_size=224, margin=40, select_largest=True, thresholds=[0.5, 0.6, 0.6])
        self.facenet = InceptionResnetV1(pretrained='vggface2', classify=False, num_classes=None).eval()

        # freeze all the face weights
        
        for param in self.facenet.parameters():
            param.requires_grad = False
        
        

        self.use_mediapipe = True
        if self.use_mediapipe:
            base_options = pythonmp.BaseOptions(model_asset_path='detector.tflite')
            options = vision.FaceDetectorOptions(base_options=base_options)
            self.face_detector = vision.FaceDetector.create_from_options(options)
        else:
            self.mtcnn = MTCNN(image_size=224, margin=40, select_largest=True)
            for param in self.mtcnn.parameters():
                param.requires_grad = False
        
        self.save_embeddings = True
        if self.save_embeddings:
            if self.use_mediapipe:
                self.log_dir = os.path.join('/home/neerja/neerja_PHALP/mediapipe_embeddings', self.cfg.video.source.split("/")[-1])
            else:
                self.log_dir = os.path.join('/home/neerja/neerja_PHALP/facenet_embeddings', self.cfg.video.source.split("/")[-1])
            os.makedirs(self.log_dir, exist_ok=True)

    
    def run_additional_models(self, image_frame, pred_bbox, pred_masks, pred_scores, pred_classes, frame_name, t_, measurments, gt_tids, gt_annots):
        # crop image_frame using pred_bbox
        crops = []
        for box in pred_bbox:
            y_min, x_min, y_max, x_max = map(int, [box[0], box[1], box[2], box[3]])
            width, height, _ = image_frame.shape
            y_min = max(0, y_min)
            x_min = max(0, x_min)
            y_max = min(height, y_max)
            x_max = min(width, x_max)
            crop = image_frame[x_min:x_max, y_min:y_max]
            crops.append(crop)

        embeds = [] 
        confs = []
        for i, crop in enumerate(crops):

            if not self.use_mediapipe:
                try:
                    img = Image.fromarray(crop)
                    mtcnn_out, conf = self.mtcnn(img, return_prob=True)
                except:
                    print("error! writing or detecting image")
                    cv2.imwrite(f"error_image_{frame_name}.jpg", image_frame)
                    img = Image.fromarray(image_frame)
                    mtcnn_out = None 
                    conf = None
                

                if conf is not None:
                    confs.append(conf)
                else:
                    confs.append(0)
                if mtcnn_out is not None:
                    img_embedding = self.facenet(mtcnn_out.unsqueeze(0))
                else:
                    # if face detector fails, set embedding to zeros
                    # crop_tensor = transforms.ToTensor()(img).to(torch.float32)
                    print("face detector failed")
                    img_embedding = torch.zeros(1, 512)
                embeds.append(img_embedding) # shape 1x512
            else:
                # import ipdb; ipdb.set_trace()
                cv_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                img = mp.Image(image_format = mp.ImageFormat.SRGB, data=cv_crop)
                detection_results = self.face_detector.detect(img)
                max_conf = 0
                face_bbox = None
                for detection in detection_results.detections:
                    bbox = detection.bounding_box
                    origin_x = bbox.origin_x
                    origin_y = bbox.origin_y
                    width = bbox.width
                    height = bbox.height
                    confidence = detection.categories[0].score
                    if confidence > max_conf:
                        max_conf = confidence
                        face_bbox = bbox
                if max_conf > 0:
                    # run detector 
                    face_image = crop[face_bbox.origin_y:face_bbox.origin_y + face_bbox.height, face_bbox.origin_x:face_bbox.origin_x + face_bbox.width]
                    face_image = cv2.resize(face_image, (160, 160))
                    face_tensor = transforms.ToTensor()(face_image).to(torch.float32)
                    img_embedding = self.facenet(face_tensor.unsqueeze(0))   
                    embeds.append(img_embedding)
                    confs.append(max_conf)
                    failed = False
                else:
                    print("face detector failed")
                    img_embedding = torch.zeros(1, 512)
                    embeds.append(img_embedding)
                    confs.append(0)
                    failed = True



            if self.save_embeddings:
                frame_name = frame_name.split('.')[0]
                frame_name = frame_name.split('/')[-1]
                with open(os.path.join(self.log_dir, frame_name + f'_{i}_emb.pkl'), 'wb') as f:
                    pickle.dump(img_embedding.detach().cpu().numpy(), f)
                with open(os.path.join(self.log_dir, frame_name + f'_{i}_img.pkl'), 'wb') as f:
                    if self.use_mediapipe:
                        if not failed:
                            pickle.dump(face_image, f)
                            cv2.imwrite(os.path.join(self.log_dir, frame_name + f'_{i}_mediapipe.jpg'), face_image)
                        else:
                            pickle.dump(crop, f)
                            cv2.imwrite(os.path.join(self.log_dir, frame_name + f'_{i}_mediapipe.jpg'), crop)
                    else:
                        if mtcnn_out is not None:
                            # undo normalization
                            mtcnn_out = mtcnn_out * 128 + 127.5
                            img = mtcnn_out.permute(1, 2, 0).detach().cpu().numpy()
                            pickle.dump(img, f)
                            # save image 
                            cv2.imwrite(os.path.join(self.log_dir, frame_name + '.jpg'), img)
                        else:
                            pickle.dump(crop, f)
                            cv2.imwrite(os.path.join(self.log_dir, frame_name + '.jpg'), crop)
        
        # return embeds, confs 
        # import ipdb; ipdb.set_trace()
        return embeds
