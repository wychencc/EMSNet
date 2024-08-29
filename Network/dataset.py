import cv2
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import torch

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


img_opt_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

img_sar_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

mask_transform = MaskToTensor()

class OurDataset(Dataset):
    def __init__(self, class_name, root, mode=None, img_sar_transform=img_sar_transform, img_opt_transform=img_opt_transform, mask_transform=mask_transform, sync_transforms=None):
        # 数据相关
        self.class_names = class_name
        self.mode = mode
        self.img_sar_transform = img_sar_transform
        self.img_opt_transform = img_opt_transform
        self.mask_transform = mask_transform
        self.sync_transform = sync_transforms
        self.sync_img_mask = []

        img_sar_dir = os.path.join(root, 'sar')
        img_opt_dir = os.path.join(root, 'opt')
        mask_dir = os.path.join(root, 'lbl')

        for img_filename in os.listdir(img_sar_dir):
            img_mask_pair = (os.path.join(img_sar_dir, img_filename),
                             os.path.join(img_opt_dir, img_filename),
                             os.path.join(mask_dir, img_filename))
            self.sync_img_mask.append(img_mask_pair)
        # print(self.sync_img_mask)

        if (len(self.sync_img_mask)) == 0:
            print("Found 0 data, please check your dataset!")

    def __getitem__(self, index):
        img_sar_path, img_opt_path, mask_path = self.sync_img_mask[index]

        img_sar2 = cv2.imread(img_sar_path, -1)
        img_sar = Image.fromarray(img_sar2)

        img_opt2 = cv2.imread(img_opt_path, -1)
        img_opt = Image.fromarray(img_opt2)

        mask2 = cv2.imread(mask_path, -1)
        mask = Image.fromarray(mask2).convert('L')

        if self.sync_transform is not None:
            img_sar, img_opt, mask = self.sync_transform(img_sar, img_opt, mask)
        if self.img_sar_transform is not None:
            img_sar = self.img_sar_transform(img_sar)
            img_opt = self.img_opt_transform(img_opt)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        return img_sar, img_opt, mask

    def __len__(self):
        return len(self.sync_img_mask)

    def classes(self):
        return self.class_names


if __name__ == "__main__":
    pass