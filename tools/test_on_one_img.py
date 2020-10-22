import os
import sys
import torch
import numpy as np
sys.path.append('.')
from config import cfg
from data.transforms import build_transforms
from modeling.baseline import Baseline
from PIL import Image


def load_image(image_name):
    image = Image.open(image_name)
    val_transforms = build_transforms(cfg, is_train=False)
    image = val_transforms(image)
    image = image.unsqueeze(0)
    return image

def load_trained_model():
    num_classes = 10
    _model = Baseline(num_classes, 1, '/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/A_PROJECTS/LOCAL/cvpr2021/LTCCscript/resnet50-19c8e357.pth', 'bnneck', 'after', 'resnet50', 'imagenet')
    _model.load_param("/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/A_PROJECTS/LOCAL/cvpr2021/LTCCscript/OUTPUT/fixcolor_N1/resnet50_model_10.pth")
    _model.eval()
    return _model

if __name__ == '__main__':

    save_feat_dir = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/shapes/train_imgs_features"
    os.makedirs(save_feat_dir, exist_ok=True)
    img_dir = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/shapes/same_colour/train"

    # prepare the model
    model = load_trained_model()

    img_names = os.listdir(img_dir)
    for i, image_name in enumerate(img_names):
        image = load_image(os.path.join(img_dir, image_name))
        with torch.no_grad():
            features = model(image)
            feature_arry = features.numpy()
            file = os.path.join(save_feat_dir,"{}.npy".format(image_name))
            np.save(file=file, arr=feature_arry)
        if i%500 == 0:
            print("feature extraction: \t{}/{}".format(i, len(img_names)))
