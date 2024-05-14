# Use pretrained model to segment

from torchvision.models.segmentation import (
    lraspp_mobilenet_v3_large,
    deeplabv3_resnet50,
    deeplabv3_mobilenet_v3_large,
)
import argparse
from loguru import logger

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import fnmatch
import numpy as np
import torchvision.transforms as transforms
import random
# import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm
import math
from models import Model4Band

import rasterio
from PIL import Image
import matplotlib.pyplot as plt
import cv2


parser = argparse.ArgumentParser(description='Lang SAM')
parser.add_argument('--fp', default="Connecticut_20230706_01.tif", type=str, help='Filepath to input tif file (expects 4 bands)')
parser.add_argument('--ckpt_path', default="deeplab_mobilenet_20240121-224805_e72.pth", type=str, help='Path to model checkpoint')
parser.add_argument('--with_transforms', default=1, type=int, help='Flag to standardize image (1 to standardize. 0 to not)')
# parser.add_argument('--thresh', default=0.4, type=float, help='Confidence threshold for predicting masks')  # tunable parameter
# parser.add_argument('--crop_size', default=768, type=int, help='Size of crops in one dimension.')   # NOTE: model predicts on smaller crops of
                                                                                                    # the full image/tile so that finer details could be processed better; 
                                                                                                    # this can be tuned if needed
opt = parser.parse_args()

EPS = 1e-7
def preprocess_input(input_chunk, trans):
    image = torch.from_numpy(input_chunk[:,:,:])
    image = image/255.   # normalize 0 to 1
    if trans is not None:
        image = trans(image)
    return image.type(torch.FloatTensor)

def compute_mask_metrics(pred, target, thresh=0.5):
    thresh_pred = np.where(pred > thresh, 1., 0.)
    rec = get_recall(thresh_pred, target)
    prec = get_precision(thresh_pred, target)
    f1 = get_f1(thresh_pred, target)

    return rec, prec, f1

def get_recall(y_pred, y_true):
    TP = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    TP_FN = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = TP / (TP_FN + EPS)
    return recall

def get_precision(y_pred, y_true):
    TP = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    TP_FP = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = TP / (TP_FP + EPS)
    return precision

def get_f1(y_pred, y_true):
    precision = get_recall(y_pred, y_true)
    recall = get_precision(y_pred, y_true)
    return 2 * ((precision * recall) / (precision + recall + EPS))


def get_prediction_from_chunks(model, crop_size, input_data, thresh):
    model.eval()
    pred_out_img = None
    ctr = 1
    for i in range(int(np.ceil(s1/crop_size))):
        vert_img = None
        for j in range(int(np.ceil(s2/crop_size))):
            ctr += 1
            input_chunk = input_data[:, i*crop_size : (i+1)*crop_size, j*crop_size : (j+1)*crop_size,]
            
            nonzero_mask = (np.sum(input_chunk, axis=0) > 0).astype(int)
            img = preprocess_input(input_chunk, trans)
            with torch.no_grad():
                pred_img = model(img.unsqueeze(0).cuda())["out"]
            pred_np = pred_img.detach().cpu().numpy().squeeze()
            pred_np = pred_np>thresh
            pred_np = pred_np * nonzero_mask   # Multiply to nonzero mask (to make sure zero pixels are not mistaken as water)
            
            if vert_img is None:
                vert_img = np.zeros_like(pred_np)
                vert_img[:] = pred_np[:]
            else:
                # logger.debug(f"vert_img: {vert_img.shape}, pred_np: {pred_np.shape}")
                vert_img = np.concatenate((vert_img, pred_np), axis=1)
        if pred_out_img is None:
            pred_out_img = vert_img
        else:
            pred_out_img = np.concatenate((pred_out_img, vert_img), axis=0)
    return pred_out_img

if __name__ == "__main__":
    opt = parser.parse_args()
    logger.debug(f"opt: {opt}")
    model_type = "deeplab_mobilenet"

    is_rgb = False
    model = Model4Band(model_type=model_type).cuda()
    # if opt.crop_size is None:
    #     crop_size = model.crop_size
    # else:
    #     crop_size = opt.crop_size
    if opt.with_transforms==1:
        trans = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406, 0.485), (0.229, 0.224, 0.225, 0.229)),
        ])
    else:
        trans = None
    logger.debug(f"Using model: {model_type}")
    ckpt = torch.load(opt.ckpt_path)
    tmp = model.load_state_dict(ckpt["state_dict"])
    logger.debug(f"Loaded model checkpoint: {tmp}")


    fp = opt.fp # filepath to the input file with 4 bands


    out_name = fp.replace(".tif", f"_pred.png")
    out_name_tif = fp.replace(".tif", f"_pred.tif")
    logger.debug(f"Processing file: {fp}. out_name: {out_name}")
    
    input_dataset = rasterio.open(fp)
    input_data = input_dataset.read()   # (4, x, y) -- (C,H,W)
    _, s1,s2 = input_data.shape

    # pred_out_img = get_prediction_from_chunks(model, crop_size, input_data, thresh=opt.thresh)
    pred_imgs = []
    for mult_crop_size in [768, 1024, 1600, 2048]:
        tmp_pred_out_img = get_prediction_from_chunks(model, mult_crop_size, input_data, thresh=0.9) # high threshold
        pred_imgs.append(tmp_pred_out_img)
    pred_out_img = np.any(np.array(pred_imgs), axis=0)
    logger.debug(f"pred_out_img: {pred_out_img.shape}")

    plt.imshow(pred_out_img, cmap="gray")
    plt.savefig(f"{out_name}.jpg")
    plt.close()

    out_3channel = np.stack((pred_out_img,pred_out_img,pred_out_img), axis=-1)
    cv2.imwrite(f"{out_name}", out_3channel*255)

    # Write to TIFF
    kwargs = input_dataset.meta
    kwargs.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw')

    with rasterio.open(out_name_tif, 'w', **kwargs) as dst:
        dst.write_band(1, pred_out_img.astype(rasterio.float32))


# TODO: try multi-scle and merge outputs
# (e.g., crop sizes 768, 1024 and merge into one output) and maybe set threshold high (~0.8 or 0.7) to only get high confidence areas