import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import json
import math

class WindowsDataset(Dataset):
    def __init__(self, images_dir, keypoints_dir, transforms=None):
        self.images_dir = images_dir
        self.keypoints_dir = keypoints_dir
        self.transfoms = transforms 
        self.images = os.listdir(images_dir)
        # A gaussian kernel cache, so we don't have to regenerate them every time.
        # This is only a small optimization, generating the kernels is pretty fast.
        self._gaussians = {}

    def __len__(self):
        return len(self.images)

    #     # apply gaussian kernel to image
    # def _gaussian(self, xL, yL, sigma, H, W):

    #     channel = [math.exp(-((c - xL) ** 2 + (r - yL) ** 2) / (2 * sigma ** 2)) for r in range(H) for c in range(W)]
    #     channel = np.array(channel, dtype=np.float32)
    #     channel = np.reshape(channel, newshape=(H, W))

    #     return channel

    # convert original image to heatmap
    def _convertToHM(self, img, keypoints, sigma=5):
        H = img.shape[1]
        W = img.shape[2]
        nKeypoints = len(keypoints)

        img_hm = np.zeros(shape=(H, W, nKeypoints), dtype=np.float32)

        for i in range(nKeypoints):
            x = keypoints[i][0]
            y = keypoints[i][1]

            channel_hm = self._generate_gaussian(x, y, H, W, sigma)
            img_hm[:, :, i] = channel_hm
        
        # img_hm = np.reshape(img_hm, newshape=(img_hm.shape[0]*img_hm.shape[1]*nKeypoints, 1))
        
        return img_hm


    def _generate_gaussian(self, x, y, h, w, sigma=5):
        """
        Generates a 2D Gaussian point at location x,y in tensor t.
        
        x should be in range (-1, 1) to match the output of fastai's PointScaler.
        
        sigma is the standard deviation of the generated 2D Gaussian.
        """
        heatmap = np.zeros((h, w))
        
        # Heatmap pixel per output pixel
        # mu_x = int(0.5 * (x + 1.) * w)
        # mu_y = int(0.5 * (y + 1.) * h)
        mu_x = int(x)
        mu_y = int(y)

        tmp_size = sigma * 3
        
        # Top-left
        x1,y1 = int(mu_x - tmp_size), int(mu_y - tmp_size)

        # Bottom right
        x2, y2 = int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)

        if x1 >= w or y1 >= h or x2 < 0 or y2 < 0:
            return t
        
        size = 2 * tmp_size + 1
        tx = np.arange(0, size, 1, np.float32)
        ty = tx[:, np.newaxis]
        x0 = y0 = size // 2
        
        # The gaussian is not normalized, we want the center value to equal 1
        g = self._gaussians[sigma] if sigma in self._gaussians \
                    else np.array((np.exp(- ((tx - x0) ** 2 + (ty - y0) ** 2) / (2 * sigma ** 2))))
        self._gaussians[sigma] = g
        
        # Determine the bounds of the source gaussian
        g_x_min, g_x_max = max(0, -x1), min(x2, w) - x1
        g_y_min, g_y_max = max(0, -y1), min(y2, h) - y1
        
        # Image range
        img_x_min, img_x_max = max(0, x1), min(x2, w)
        img_y_min, img_y_max = max(0, y1), min(y2, h)

        heatmap[img_y_min:img_y_max, img_x_min:img_x_max] = g[g_y_min:g_y_max, g_x_min:g_x_max]

        return heatmap
    
    def __getitem__(self, index):
        img_path = os.path.join(self.images_dir, self.images[index])
        keypoints_path = os.path.join(self.keypoints_dir, self.images[index].replace("jpg", "json"))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image) / 255.0

        # keypoints
        with open(keypoints_path) as f:
            kps = json.load(f)['points']

        if self.transfoms is not None:
            augmentations = self.transfoms(image=image, keypoints=kps)
            image = augmentations['image']
            kps = augmentations['keypoints']
            heatmap = self._convertToHM(image, kps)

        return image, heatmap

    