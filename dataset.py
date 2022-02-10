import os
from io import BytesIO
from glob import glob
import logging
import random
import numpy as np
from PIL import Image
from scipy.stats.mstats import gmean
import rawpy
import cv2
import torch
from torch.utils.data import Dataset


ev_list = [-1, 0, 1]
rawimg_extension = '/*.dng'


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, patch_size):
        self.imgs_dir = imgs_dir
        self.patch_size = patch_size

        self.imgfiles = [p for p in glob(self.imgs_dir + rawimg_extension) if os.path.isfile(p)]
        logging.info(f'Creating dataset with {len(self.imgfiles)} images')

    def __len__(self):
        return len(self.imgfiles)

    @classmethod
    def preprocess(cls, img, patch_size, patch_coords, flip_op, ev=0):
        img_patch = img[patch_coords[1]:patch_coords[1] + patch_size, 
                        patch_coords[0]:patch_coords[0] + patch_size, 
                        :]

        if ev == 0:
            img_patch, wb_patch = generate_img(img_patch)
        else:
            img_patch, wb_patch = generate_img(
                np.minimum(np.maximum((img_patch * (2**ev)), 0), 1))

        if flip_op == 1:
            img_patch = np.flip(img_patch, axis=2)
        elif flip_op == 2:
            img_patch = np.transpose(img_patch, (1, 0, 2))

        # HWC to CHW
        img_trans = img_patch.transpose((2, 0, 1))
        return img_trans, wb_patch

    def __getitem__(self, i):
        # get images
        img_file = self.imgfiles[i]
        img_input_raw = cv2.resize(readraw_demosaicing_wb_srgb(img_file), None, fx=0.5, fy=0.5)
        img_input_raw = norm_lum(img_input_raw)

        # get image size
        w, h = img_input_raw.shape[1], img_input_raw.shape[0]

        # get flipping option
        flip_op = np.random.randint(3)
        # get random patch coord
        patch_x = np.random.randint(0, high=w - self.patch_size)
        patch_y = np.random.randint(0, high=h - self.patch_size)

        img_in_patchs, wb_patch = self.preprocess(img_input_raw, 
                                                self.patch_size, 
                                                (patch_x, patch_y), 
                                                flip_op, 
                                                ev=0)
        tmp_img, tmp_wb = self.preprocess(img_input_raw, 
                                        self.patch_size, 
                                        (patch_x, patch_y), 
                                        flip_op, 
                                        ev=-1)

        img_in_patchs = np.append(img_in_patchs, tmp_img, axis=0)
        wb_patch = np.append(wb_patch, tmp_wb, axis=0)

        tmp_img, tmp_wb = self.preprocess(img_input_raw, 
                                        self.patch_size, 
                                        (patch_x, patch_y), 
                                        flip_op, 
                                        ev=1)

        img_in_patchs = np.append(img_in_patchs, tmp_img, axis=0)
        wb_patch = np.append(wb_patch, tmp_wb, axis=0)

        return {'input': torch.from_numpy(img_in_patchs),
                'input_wb': torch.from_numpy(wb_patch)}


def readraw_demosaicing_wb_srgb(fname):
    with rawpy.imread(fname) as raw:
        raw_rgb = raw.postprocess(
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.sRGB,
            gamma=(1., 1.),  # no apply gamma correction
            output_bps=16,
            no_auto_bright=True
        )
    return raw_rgb.astype(np.float64)/65535


def gamma_correction(raw_rgb, gamma=2.2):
    raw_rgb_gamma = np.maximum(raw_rgb, 1e-8) ** (1.0 / gamma)
    return np.minimum(np.maximum(raw_rgb_gamma, 0), 1)


def generate_img(img_raw_rgb):
    img_changed_ratio, wb_new = wb_change(img_raw_rgb)
    buffer = BytesIO()
    Image.fromarray((img_changed_ratio * 255).astype(np.uint8)).save(buffer, "PNG")
    img_input = np.asarray(Image.open(buffer)).astype(np.float32) / 255
    return img_input, wb_new


def tonemap_smoothstep(raw_rgb_gamma):
    img_tm = (3 * (raw_rgb_gamma**2)) - (2 * (raw_rgb_gamma**3))
    return np.minimum(np.maximum(img_tm, 0), 1)


def luminance_calculation(img):
    return 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]


def norm_lum(img, alpha=0.18):
    lum = luminance_calculation(img)
    lum_gmean = gmean(lum.flatten() + 1e-6) 
    lum_norm = (alpha/lum_gmean) * lum
    lum_3d = np.repeat(lum[:, :, np.newaxis], 3, axis=2)
    lum_norm_3d = np.repeat(lum_norm[:, :, np.newaxis], 3, axis=2)
    img_norm = (img+1e-6) / (lum_3d+1e-6) * lum_norm_3d
    return np.minimum(np.maximum(img_norm, 0), 1)


def wb_change(img, a=0.9, b=1.1):
    lum_tmp = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]
    lum = np.repeat(lum_tmp[:, :, np.newaxis], 3, axis=2)

    change_lum = np.zeros(img.shape)
    wb_new = np.zeros([3])
    for ch in range(img.shape[2]):
        wb_new[ch] = random.choice([a, 1.0, b]) # a, 1.0 or b
        change_lum[:,:,ch] = lum_tmp * wb_new[ch]

    new_img = img / np.maximum(lum,1e-08) * change_lum
    new_lum = 0.2126*new_img[:,:,0] + 0.7152*new_img[:,:,1] + 0.0722*new_img[:,:,2]
    new_img = new_img / np.maximum(np.repeat(new_lum[:, :, np.newaxis], 3, axis=2), 1e-08) * lum
    return new_img/np.max(new_img), wb_new
