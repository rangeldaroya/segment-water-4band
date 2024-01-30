from torchvision.models.segmentation import (
    lraspp_mobilenet_v3_large,
    deeplabv3_resnet50,
    deeplabv3_mobilenet_v3_large,
)
import argparse
from loguru import logger
import torch.optim as optim
from torch.utils.data.dataset import Dataset
import torchmetrics
from torchmetrics.classification import BinaryJaccardIndex

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import fnmatch
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import mobilenetv3, ResNet50_Weights
import random
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm
from datetime import datetime
import math
from models import Model4Band


EPS = 1e-7  # for max pool loss and metrics (f1, recall, precision)
DATE_STR = datetime.now().strftime("%Y%m%d-%H%M%S")
parser = argparse.ArgumentParser(description='Lang SAM')
parser.add_argument('--ckpt_dir', default="ckpts", type=str, help='Checkpoint directory')
parser.add_argument('--model_type', default="lraspp", type=str, help='Type of model to use. Choices: [deeplab_mobilenet, deeplab_resnet, lraspp]')
parser.add_argument('--input_type', default="rgb", type=str, help='Type of input to use. Choices: [rgb, 4band]')
parser.add_argument('--loss_type', default="bce", type=str, help='Type of loss to use. Choices: [bce, adapmaxpool]')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate to use')
# parser.add_argument('--thresh', default=0.35, type=float, help='Confidence threshold')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
parser.add_argument('--num_epochs', default=100, type=int, help='Num of epochs for training')

# TODO: make a 4-band version (standardize img values), and a 3-band version

class SatelliteDataRGB(Dataset):
    """
    This file is directly modified from https://pytorch.org/docs/stable/torchvision/datasets.html
    """
    def __init__(self, root, crop_size, split='train', transforms=None, is_rgb=False):
        self.split = split
        self.root = os.path.expanduser(root)
        self.transforms = transforms
        self.is_rgb = is_rgb

        # R\read the data file
        if split=="train":
            self.data_path = os.path.join(root, 'train')
        elif split=="val":
            self.data_path = os.path.join(root, 'val')
        elif split=="test":
            self.data_path = os.path.join(root, 'test')
        else:
            raise NotImplementedError

        # calculate data length
        self.fps = fnmatch.filter(os.listdir(self.data_path), '*.npy')
        self.data_len = len(self.fps)
        self.crop_size = crop_size

    def __getitem__(self, index):
        # fp = os.path.join(self.data_path, f"{index:06d}.npy")
        fp = os.path.join(self.data_path, self.fps[index])
        with open(fp, "rb") as f:
            npzfile = np.load(f)
            input_data = npzfile["input"]  # (4, 256, 256) (C,H,W)
            output_data = npzfile["output"]  # (1, 256, 256) (C,H,W)
        
        label = torch.from_numpy(output_data)
        if self.is_rgb:
            image = torch.from_numpy(input_data[:3,:,:])
            if self.transforms is not None:
                image = image[:, :self.crop_size, :self.crop_size]
                label = label[:, :self.crop_size, :self.crop_size]
                image = self.transforms(image)
        else:
            image = torch.from_numpy(input_data[:,:,:])
            # if self.transforms is not None:
            image = image[:, :self.crop_size, :self.crop_size]
            label = label[:, :self.crop_size, :self.crop_size]
            image = image/255.   # normalize 0 to 1
            image = self.transforms(image)
            # img_mean = torch.reshape(torch.mean(image, dim=(1,2)), (4,1,1))
            # img_var = torch.reshape(torch.var(image, dim=(1,2)), (4,1,1))
            # image = (image - img_mean)/(img_var**0.5)     # standardize
        return (
            image.type(torch.FloatTensor),
            label.type(torch.FloatTensor),
        )

    def __len__(self):
        return self.data_len

def get_model_loaders(opt):
    # For pretrained models, see https://pytorch.org/vision/stable/models.html#semantic-segmentation
    if opt.input_type == "rgb":
        is_rgb = True
        if opt.model_type == "deeplab_mobilenet":
            model = deeplabv3_mobilenet_v3_large(
                weights_backbone="IMAGENET1K_V1",
                num_classes=1,
            ).cuda()
            trans = mobilenetv3.MobileNet_V3_Large_Weights.IMAGENET1K_V1.transforms()
        elif opt.model_type == "deeplab_resnet":
            model = deeplabv3_resnet50(
                weights_backbone="IMAGENET1K_V1",
                num_classes=1,
            ).cuda()
            trans = ResNet50_Weights.IMAGENET1K_V1.transforms()
        elif opt.model_type == "lraspp":
            model = deeplabv3_resnet50(
                weights_backbone="IMAGENET1K_V1",
                num_classes=1,
            ).cuda()
            trans = mobilenetv3.MobileNet_V3_Large_Weights.IMAGENET1K_V1.transforms()
        else:
            raise NotImplementedError
        crop_size = trans.__dict__["crop_size"][0]
    elif opt.input_type == "4band":
        is_rgb = False
        model = Model4Band(model_type=opt.model_type).cuda()
        crop_size = model.crop_size
        trans = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406, 0.485), (0.229, 0.224, 0.225, 0.229)),
        ])
    else:
        raise NotImplementedError

    train_dataset = SatelliteDataRGB(root="segtrain_data", split="train", crop_size=crop_size, transforms=trans, is_rgb=is_rgb)
    val_dataset = SatelliteDataRGB(root="segtrain_data", split="val", crop_size=crop_size, transforms=trans, is_rgb=is_rgb)
    test_dataset = SatelliteDataRGB(root="segtrain_data", split="test", crop_size=crop_size, transforms=trans, is_rgb=is_rgb)
    return model, train_dataset, val_dataset, test_dataset


class MaxPool2dSame(torch.nn.MaxPool2d):
    # Since pytorch Conv2d does not support same padding the same way in Keras, had to do a workaround
    # Adopted from https://github.com/pytorch/pytorch/issues/67551#issuecomment-954972351
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size, s=self.stride, d=self.dilation)
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size, s=self.stride, d=self.dilation)

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.max_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
def adaptive_maxpool_loss(y_pred, y_true, alpha=0.25):
    # logger.debug(f"y_pred: {y_pred.shape}, y_true: {y_true.shape}")
    # From https://github.com/isikdogan/deepwatermap/blob/master/metrics.py#L33-L41
    y_pred, y_true = torch.squeeze(y_pred,1), torch.squeeze(y_true,1)
    y_pred = torch.clip(y_pred, EPS, 1-EPS)
    positive = -y_true * torch.log(y_pred) * alpha
    negative = -(1. - y_true) * torch.log(1. - y_pred) * (1-alpha)
    pointwise_loss = positive + negative
    max_loss = MaxPool2dSame(kernel_size=8, stride=1)(pointwise_loss)
    x = pointwise_loss * max_loss
    x = torch.mean(x, dim=1)   # channel is index 1
    return torch.mean(x)

def get_miou(preds, gts, thresh=0.5):
    metric_fn = BinaryJaccardIndex(threshold=thresh).cuda()
    return metric_fn(preds, gts)

def compute_mask_metrics(pred, target, thresh=0.5):
    pred = torch.squeeze(pred, 1)
    target = torch.squeeze(target, 1)
    thresh_pred = torch.where(pred > thresh, 1., 0.)
    rec = get_recall(thresh_pred, target)
    prec = get_precision(thresh_pred, target)
    f1 = get_f1(thresh_pred, target)

    return rec, prec, f1

def get_recall(y_pred, y_true):
    with torch.no_grad():
        TP = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
        TP_FN = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
        recall = TP / (TP_FN + EPS)
    return recall

def get_precision(y_pred, y_true):
    with torch.no_grad():
        TP = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
        TP_FP = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
        precision = TP / (TP_FP + EPS)
    return precision

def get_f1(y_pred, y_true):
    with torch.no_grad():
        precision = get_recall(y_pred, y_true)
        recall = get_precision(y_pred, y_true)
    return 2 * ((precision * recall) / (precision + recall + EPS))

def train(model, train_loader, optimizer, loss_fn):
    model.train()
    train_loss = 0
    # train_metric = 0
    train_rec, train_prec, train_f1 = 0,0,0
    for idx, (img, label) in enumerate(train_loader):
        img, label = img.cuda(), label.cuda()
        pred = model(img)["out"]
        loss = loss_fn(pred, label)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            rec, prec, f1 = compute_mask_metrics(pred, label)
            train_rec, train_prec, train_f1 = train_rec+rec, train_prec+prec, train_f1+f1
            # train_metric += get_miou(pred, label)
    # ave_loss, ave_metric = train_loss/(idx + 1), train_metric/(idx + 1)
    ave_loss = train_loss/(idx + 1)
    ave_rec, ave_prec, ave_f1 = train_rec/(idx+1), train_prec/(idx+1), train_f1/(idx+1)
    return ave_loss, ave_rec, ave_prec, ave_f1

def evaluate(model, val_loader, loss_fn):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        # val_metric = 0
        val_rec, val_prec, val_f1 = 0,0,0
        for idx, (img, label) in enumerate(val_loader):
            img, label = img.cuda(), label.cuda()
            pred = model(img)["out"]
            loss = loss_fn(pred, label)
            val_loss += loss.item()

            rec, prec, f1 = compute_mask_metrics(pred, label)
            val_rec, val_prec, val_f1 = val_rec+rec, val_prec+prec, val_f1+f1
            # val_metric += get_miou(pred, label)
    # ave_loss, ave_metric = val_loss/(idx + 1), val_metric/(idx + 1)
    ave_loss = val_loss/(idx + 1)
    ave_rec, ave_prec, ave_f1 = val_rec/(idx+1), val_prec/(idx+1), val_f1/(idx+1)
    return ave_loss, ave_rec, ave_prec, ave_f1


if __name__ == "__main__":
    opt = parser.parse_args()
    logger.debug(f"opt: {opt}")

    if not os.path.isdir(opt.ckpt_dir):
        os.makedirs(opt.ckpt_dir, exist_ok=True)
    model_ckpt_path = os.path.join(opt.ckpt_dir, opt.model_type)
    if not os.path.isdir(model_ckpt_path):
        os.makedirs(model_ckpt_path, exist_ok=True)

    model, train_dataset, val_dataset, test_dataset = get_model_loaders(opt)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=opt.batch_size,
        shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=opt.batch_size,
        shuffle=False, num_workers=0)
    
    if opt.loss_type == "bce":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif opt.loss_type == "adapmaxpool":
        loss_fn = adaptive_maxpool_loss

    start_epoch = 0
    for epoch in range(start_epoch, opt.num_epochs):
        train_loss, train_rec, train_prec, train_f1 = train(model, train_loader, optimizer, loss_fn)
        logger.debug(f"[{epoch:02d}/{opt.num_epochs}] TRAIN loss: {train_loss}, metrics: {train_f1}, {train_rec}, {train_prec}")
        val_loss, val_rec, val_prec, val_f1 = evaluate(model, val_loader, loss_fn)
        logger.debug(f"[{epoch:02d}/{opt.num_epochs}] VAL loss: {val_loss}, metrics: {val_f1}, {val_rec}, {val_prec}")
        test_loss, test_rec, test_prec, test_f1 = evaluate(model, test_loader, loss_fn)
        logger.debug(f"[{epoch:02d}/{opt.num_epochs}] TEST loss: {test_loss}, metrics: {test_f1}, {test_rec}, {test_prec}")

        filepath = os.path.join(model_ckpt_path, f'{DATE_STR}_e{epoch:02d}.pth')
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'lr': opt.lr,
            'input_type': opt.input_type,
            'batch_size': opt.batch_size,
        }, filepath)
