import rasterio
import numpy as np
from lang_sam import LangSAM
from PIL import Image
from loguru import logger
import matplotlib.pyplot as plt
import cv2
import argparse


parser = argparse.ArgumentParser(description='Lang SAM')
parser.add_argument('--i', default=0, type=int, help='Index to process [0-31]')
opt = parser.parse_args()

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

# fp = "Connecticut_20230706_01.tif"
fp = all_fps[opt.i]
out_name = fp.replace(".tif", ".png")
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



# TODO: Process samp_img here (pred_img is the water mask)
nonzero_mask = (np.sum(img, axis=-1) > 0).astype(int)
image_pil = Image.fromarray(np.uint8(img))
text_prompt = "river"
masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
logger.debug(f"boxes: {boxes}")
logger.debug(f"phrases: {phrases}")
logger.debug(f"logits: {logits}")
pred_img = masks.detach().numpy().squeeze().astype(int)
logger.debug(f"pred_img: {pred_img} {pred_img.shape}")
pred_shape = pred_img.shape
if len(pred_img) < 1:
    s1, s2, _ = img.shape
    pred_img = np.zeros((s1,s2))
    logger.debug(f"No mask. pred_img: {pred_img.shape}")
elif len(pred_shape) == 3:  # means there are multiple masks
    pred_img = (np.sum(pred_img, axis=0) > 0).astype(int)   # combine all masks
    logger.debug(f"Found mulltiple masks. pred_shape: {pred_shape}. pred_img: {pred_img.shape}")
pred_img = pred_img * nonzero_mask   # Multiply to nonzero mask (to make sure zero pixels are not mistaken as water)

plt.imshow(pred_img, cmap="gray")
plt.savefig(f"out_img_whole.png")
plt.close()

out_3channel = np.stack((pred_img,pred_img,pred_img), axis=-1)
cv2.imwrite(f"{out_name}", out_3channel*255)
