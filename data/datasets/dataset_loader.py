# encoding: utf-8
"""
@author:  liaoxingyu (read_image and ImageDataset)
@contact: sherlockliao01@gmail.com

@author:  Ehsan Yaghoubi (ReceptiveFieldEnhancerImageDataset)
@contact: Ehsan.Yaghoubi@gmail.com

"""

import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from data.transforms.customized_data_augmentation import PartSubstitution
import numpy as np

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    img = None
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, feat2_path, clothid = self.dataset[index]
        img = read_image(img_path)
        if feat2_path is not None:
            try:
                feat2 = np.load(feat2_path)
            except FileNotFoundError:
                # print("features for this image is not found: {}".format(feat2_path))
                feat2 = np.ones((1,2048), dtype=np.float32)
        else:
            feat2 = None

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path, feat2, int(clothid)


class ReceptiveFieldEnhancerImageDataset(Dataset):  # RFE: Receptive Field Enhancer
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform, is_train, online_image_processing_for_each_image, online_image_processing_for_all_images_once, MaskDir=None, ImgDir=None, swap_roi_rou=None, target_background_dir=None, probability=None, TargetImagesArray= None):
        self.dataset = dataset
        self.transform = transform
        self.MaskDir = MaskDir
        self.ImgDir = ImgDir
        self.is_train = is_train
        self.swap_roi_rou = swap_roi_rou
        self.probability = probability
        self.target_background_dir = target_background_dir
        self. online_image_processing_for_each_image = online_image_processing_for_each_image
        self.online_image_processing_for_all_images_once = online_image_processing_for_all_images_once
        self.image_obj = PartSubstitution(probability=probability,
                                          MaskDir = MaskDir,
                                          ImgDir = ImgDir,
                                          target_background_dir = target_background_dir,
                                          online_image_processing_for_each_image=online_image_processing_for_each_image,
                                          online_image_processing_for_all_images_once=online_image_processing_for_all_images_once,
                                          constraint_funcs=False, other_attrs=None, TargetImagesArray = TargetImagesArray)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # img_path, pid, camid, img_labels = self.dataset[index]
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)  # img is read as PIL image
        if self.transform is not None:
            if self.is_train:
                if self.swap_roi_rou:  # Augment the data with exchanging the region of interest with unwanted region?
                    while True:
                        imge = self.image_obj(current_image_path=img_path)
                        if imge is not None:
                            break

                    if not isinstance(imge, Image.Image):  # convert the image type to PIL if it is not already
                        img = Image.fromarray(imge, 'RGB')
                    else:
                        img = imge

                img = self.transform(img)  # in this line **train** transformation (other augmentations) is applied

            else:  # validation of test phase
                if self.swap_roi_rou:  # Augment the data with exchanging the region of interest with region of uninterest?
                    while True:
                        imge = self.image_obj(current_image_path=img_path)
                        if imge is not None:
                            break

                    if not isinstance(imge, Image.Image):  # convert the image type to PIL if it is not already
                        img = Image.fromarray(imge, 'RGB')
                    else:
                        img = imge

                img = self.transform(img)  # in this line **val** transformation is applied

        return img, pid, camid, img_path
