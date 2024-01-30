# Upon creation of environment, install pytorch:
# conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
import rasterio
import numpy as np
from lang_sam import LangSAM    # See https://github.com/luca-medeiros/lang-segment-anything
from PIL import Image
from loguru import logger
import matplotlib.pyplot as plt
import cv2
import argparse


parser = argparse.ArgumentParser(description='Lang SAM')
parser.add_argument('--i', default=0, type=int, help='Index to process [0-31]')
parser.add_argument('--text_prompt', default="river", type=str, help='Text prompt for segmentation')
parser.add_argument('--thresh', default=0.35, type=float, help='Confidence threshold')
parser.add_argument('--num_split', default=4, type=int, help='Num of splits in one dimension')
opt = parser.parse_args()
logger.debug(f"opt: {opt}")

all_fps = ['data/Connecticut_UTM18/Connecticut_20230706_01.tif',
 'data/Connecticut_UTM18/Connecticut_20230706_06.tif',
 'data/Connecticut_UTM18/Connecticut_20230706_13.tif',
 'data/PNW_UTM10/PNW_20230612_1_1.tif',
 'data/PNW_UTM10/PNW_20230612_1_3.tif',
 'data/PNW_UTM10/PNW_20230612_3_4.tif',
 'data/Willamette_UTM10/Willamette_20230606_01.tif',
 'data/Willamette_UTM10/Willamette_20230606_02.tif',
 'data/Willamette_UTM10/Willamette_20230606_03.tif',
 'data/Willamette_UTM10/Willamette_20230606_04.tif',
 'data/Willamette_UTM10/Willamette_20230606_06.tif',
 'data/Willamette_UTM10/Willamette_20230606_07.tif',
 'data/Willamette_UTM10/Willamette_20230606_08.tif',
 'data/Willamette_UTM10/Willamette_20230606_09.tif',
 'data/Willamette_UTM10/Willamette_20230612_01.tif',
 'data/Willamette_UTM10/Willamette_20230612_02.tif',
 'data/Willamette_UTM10/Willamette_20230612_03.tif',
 'data/Willamette_UTM10/Willamette_20230612_04.tif',
 'data/Willamette_UTM10/Willamette_20230612_05.tif',
 'data/Willamette_UTM10/Willamette_20230612_06.tif',
 'data/Willamette_UTM10/Willamette_20230612_07.tif',
 'data/Willamette_UTM10/Willamette_20230612_08.tif',
 'data/Willamette_UTM10/Willamette_20230612_09.tif',
 'data/Willamette_UTM10/Willamette_20230621_01.tif',
 'data/Willamette_UTM10/Willamette_20230621_02.tif',
 'data/Willamette_UTM10/Willamette_20230621_03.tif',
 'data/Willamette_UTM10/Willamette_20230621_04.tif',
 'data/Willamette_UTM10/Willamette_20230621_05.tif',
 'data/Willamette_UTM10/Willamette_20230621_06.tif',
 'data/Willamette_UTM10/Willamette_20230621_07.tif',
 'data/Willamette_UTM10/Willamette_20230621_08.tif',
 'data/Willamette_UTM10/Willamette_20230621_09.tif']

text_prompt = opt.text_prompt
THRESH = opt.thresh
num_x_split = opt.num_split
num_y_split = opt.num_split
# fp = "Connecticut_20230706_01.tif"
fp = all_fps[opt.i]
out_name = fp.replace(".tif", f"_{text_prompt}_{THRESH}_{num_x_split}x{num_y_split}.png")
logger.debug(f"Processing file: {fp}. out_name: {out_name}")
dataset = rasterio.open(fp)
data = dataset.read()

img = np.transpose(data, (1,2,0))[:,:,:3]


# im1 = img[:3500, :3400]
# im2 = img[3500:7000, 3400:6800]
# im3 = img[7000:10500, 6800:10200]
# im4 = img[10500:, 10200:]

# text_prompt = "water"
# image_pil = Image.fromarray(np.uint8(im4))

logger.debug("Loading model")
model = LangSAM()

x_splits = np.array_split(img, num_x_split, axis=0)
ctr = 1
out_img = None
for x_split in x_splits:    # Split big image into 16 smaller images. Then segment each one, and concatenate into one big image
    xy_splits = np.array_split(x_split, num_y_split, axis=1)
    vert_img = None
    for samp_img in xy_splits:
        logger.debug(f"ctr: {ctr}, samp_img: {samp_img.shape}")
        ctr += 1

        # TODO: Process samp_img here (pred_img is the water mask)
        nonzero_mask = (np.sum(samp_img, axis=-1) > 0).astype(int)
        image_pil = Image.fromarray(np.uint8(samp_img))
        masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
        logger.debug(f"boxes: {boxes}")
        logger.debug(f"phrases: {phrases}")
        logger.debug(f"logits: {logits}")
        pred_img = masks.detach().numpy().squeeze().astype(int)
        logger.debug(f"pred_img: {pred_img} {pred_img.shape}")
        pred_shape = pred_img.shape
        s1, s2, _ = samp_img.shape
        if len(pred_img) < 1:
            pred_img = np.zeros((s1,s2))
            logger.debug(f"No mask. pred_img: {pred_img.shape}")
        elif len(pred_shape) == 3:  # means there are multiple masks
            # pred_img = (np.sum(pred_img, axis=0) > 0).astype(int)   # combine all masks
            all_masks = np.zeros((s1,s2))
            for c in range(pred_shape[0]):
                pred_i = pred_img[c,:,:]
                if np.sum(pred_i*(1-nonzero_mask)) > 0:  # means has intersection with zero pixels
                    continue
                if logits[c] < THRESH:
                    continue
                all_masks += pred_i
            pred_img = (all_masks > 0).astype(int)
            logger.debug(f"Found mulltiple masks. pred_shape: {pred_shape}. pred_img: {pred_img.shape}")
        elif logits[0] < THRESH:
            pred_img = np.zeros((s1,s2))
            logger.debug(f"Logits below thresh: {logits}")

        pred_img = pred_img * nonzero_mask   # Multiply to nonzero mask (to make sure zero pixels are not mistaken as water)

        if vert_img is None:
            vert_img = np.zeros_like(pred_img)
            vert_img[:] = pred_img[:]
        else:
            logger.debug(f"vert_img: {vert_img.shape}, pred_img: {pred_img.shape}")
            vert_img = np.concatenate((vert_img, pred_img), axis=1)
    if out_img is None:
        out_img = vert_img
    else:
        out_img = np.concatenate((out_img, vert_img), axis=0)
    
    plt.imshow(out_img, cmap="gray")
    plt.savefig(f"out_img_{ctr}.png")
    plt.close()

plt.imshow(out_img, cmap="gray")
plt.savefig(f"{out_name}.jpg")
plt.close()

out_3channel = np.stack((out_img,out_img,out_img), axis=-1)
cv2.imwrite(f"{out_name}", out_3channel*255)
