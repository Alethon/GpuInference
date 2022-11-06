import math
from operator import index
import os
import cv2
import numpy as np
import random
from time import time
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, int32
from numpy import ndarray
import torchvision.transforms.functional as TF

import re

from utils import *

CLASS_NAMES = os.path.join('.', 'data', 'coco.names')

COCO = os.path.join('.', 'coco')
COCO_SUBSETS = os.path.join(COCO, 'subsets')

COCO_IMAGES = os.path.join(COCO, 'images')
COCO_IMAGES_TRAIN = os.path.join(COCO_IMAGES, 'train2014')
COCO_IMAGES_VAL = os.path.join(COCO_IMAGES, 'val2014')

COCO_LABELS = os.path.join(COCO, 'labels')
COCO_LABELS_TRAIN = os.path.join(COCO_LABELS, 'train2014')
COCO_LABELS_VAL = os.path.join(COCO_LABELS, 'val2014')

COCO_SUBSETS_UNROOT = os.path.join('coco', 'subsets')
COCO_LABELS_UNROOT = os.path.join('coco', 'labels')
COCO_IMAGES_UNROOT = os.path.join('coco', 'images')

IMAGES_VAL = os.path.join('images', 'val2014')

WEIGHTS = os.path.join('.', 'weights')
WEIGHTS_TINY = os.path.join('.', 'weights_tiny')

def random_affine(img, targets=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    border = 0  # width of added border (optional)
    height = max(img.shape[0], img.shape[1]) + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(height, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    if targets is not None:
        if len(targets) > 0:
            n = targets.shape[0]
            points = targets[:, 1:5].copy()
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            np.clip(xy, 0, height, out=xy)
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            targets = targets[i]
            targets[:, 1:5] = xy[i]

        return imw, targets, M
    else:
        return imw

def readDatasetInfo(path: str) -> dict[str, any]:
    data: dict[str, str] = dict()
    if os.path.exists(path) and os.path.isfile(path):
        with open(path, 'r') as f:
            data = { s[0].strip(): s[1].strip() for s in [l.split('=') for l in f.read().strip().splitlines()] if len(s) > 1 }
    result: dict[str, any] = dict()
    if 'classes' in data:
        result['classes'] = int(data['classes'])
    result['useTiny'] = True if 'useTiny' in data and data['useTiny'] == '1' else False
    result['dataPath'] = os.path.join(*data['dataPath'].split(',')) if 'dataPath' in data else os.path.join('.', 'Lego')
    result['names'] = data['names'] if 'names' in data else 'data/obj.names'
    return result

def getIntStrLabels(data: list[str]) -> list[tuple[int, str]]:
    return [(int(s[:i]), s[i + 1:]) for i, s in [(l.find(' '), l) for l in data]]

class CocoSubset:
    def __init__(self, imgSize, classList: list[str] = []) -> None:
        self.imgSize = imgSize

        with open(CLASS_NAMES, 'r') as f:
            ogClassList: list[str] = f.read().strip().split()
        
        ogIndexLookup: dict[str, int] = { cls: i for i, cls in enumerate(ogClassList) }

        if not isinstance(classList, list):
            classList = ogClassList

        classList = [cls for cls in classList if cls in ogClassList]

        if len(classList) == 0:
            classList = ogClassList
        
        self.nC: int = len(classList)
        self.classList: list[str] = classList

        self.alias: dict[int, int] = { ogIndexLookup.get(cls): i for i, cls in enumerate(self.classList) if cls in ogIndexLookup }
        self.subsetAlias: str = '-'.join([str(ogIndexLookup.get(cls)) for cls in self.classList])

        self.subset: bool = (len(self.classList) != len(ogClassList) or any(c1 != c2 for c1, c2 in zip(self.classList, ogClassList)))

        if self.subset:
            self.setDir: str = os.path.join(COCO_SUBSETS, self.subsetAlias)
            self.setDirUnroot: str = os.path.join(COCO_SUBSETS_UNROOT, self.subsetAlias)
        else:
            self.setDir: str = COCO
            self.setDirUnroot: str = 'coco'
        
        self.labelsDir: str = os.path.join(self.setDir, 'labels')
        self.labelsDirUnroot: str = os.path.join(self.setDirUnroot, 'labels')
        self.trainvalno5k: str = os.path.join(self.setDir, 'trainvalno5k.txt')
        self.k5: str = os.path.join(self.setDir, '5k.txt')

        if self.subset:
            dirsExisted: bool = self._makeDirs()
            self._makeLabels(dirsExisted)
        
        self.images: list = []
        self.labels: list[Tensor] = []
    
    def _makeDirs(self) -> bool:
        result: bool = True
        if not os.path.isdir(COCO_SUBSETS):
            os.mkdir(COCO_SUBSETS)
            result = False
        if not os.path.isdir(self.setDir):
            os.mkdir(self.setDir)
            result = False
        if not os.path.isdir(self.labelsDir):
            os.mkdir(self.labelsDir)
            result = False
        if not os.path.isdir(os.path.join(self.labelsDir, 'train2014')):
            os.mkdir(os.path.join(self.labelsDir, 'train2014'))
            result = False
        if not os.path.isdir(os.path.join(self.labelsDir, 'val2014')):
            os.mkdir(os.path.join(self.labelsDir, 'val2014'))
            result = False
        return result

    def _makeLabels(self, dirsExisted: bool) -> None:
        if dirsExisted and os.path.isfile(self.k5) and os.path.isfile(self.trainvalno5k):
            return
        with open(os.path.join(COCO, '5k.txt'), 'r') as f:
            ilPairs = [(i, i.replace('images', 'labels').replace('.png\n', '.txt').replace('.jpg\n', '.txt')) for i in f.readlines()]
        keep5k: list[tuple[str, str, str]] = []
        for imageFile, labelFile in ilPairs:
            if os.path.isfile(labelFile):
                with open(labelFile, 'r') as f:
                    labels: list[tuple[int, str]] = getIntStrLabels(f.readlines())
                labelStrings: list[str] = [str(ai) + ' ' + ss for ai, ss in [(self.alias.get(i), s) for i, s in labels] if ai is not None]
                if len(labelStrings) > 0:
                    keep5k.append((imageFile, labelFile.replace(COCO_LABELS_UNROOT, os.path.join(COCO_SUBSETS_UNROOT, self.subsetAlias, 'labels')), ''.join(labelStrings)))
        with open(os.path.join(COCO, 'trainvalno5k.txt'), 'r') as f:
            ilPairs = [(i, i.replace('images', 'labels').replace('.png\n', '.txt').replace('.jpg\n', '.txt')) for i in f.readlines()]
        keep: list[tuple[str, str, str]] = []
        for imageFile, labelFile in ilPairs:
            with open(labelFile, 'r') as f:
                labels: list[tuple[int, str]] = getIntStrLabels(f.readlines())
            labelStrings: list[str] = [str(ai) + ' ' + ss for ai, ss in [(self.alias.get(i), s) for i, s in labels] if ai is not None]
            if len(labelStrings) > 0:
                keep.append((imageFile, labelFile.replace(COCO_LABELS_UNROOT, os.path.join(COCO_SUBSETS_UNROOT, self.subsetAlias, 'labels')), ''.join(labelStrings)))
        skipno5k: list[int] = []
        for i, (imagePath, _, _) in enumerate(keep):
            if imagePath.find(IMAGES_VAL) != -1:
                skipno5k.append(i)
        missing5k: int = 5000 - len(keep5k)
        if len(skipno5k) > missing5k:
            random.seed(time())
            skipno5k = random.choices(skipno5k, k=missing5k)
        with open(self.k5, 'w+') as f5k, open(self.trainvalno5k, 'w+') as fno5k:
            n: int = 0
            for imagePath, labelPath, label in keep5k:
                with open(labelPath, 'w+') as f:
                    f.write(label)
                f5k.write(imagePath)
            for i, (imagePath, labelPath, label) in enumerate(keep):
                with open(labelPath, 'w+') as f:
                    f.write(label)
                if i == skipno5k[n]:
                    n += 1
                    f5k.write(imagePath)
                else:
                    fno5k.write(imagePath)

class LegoSegment:
    def __init__(self, segmentPath: str, segmentName: str, image: Tensor, label: ndarray, shape: tuple[float, float],
                 segmentSize: tuple[int, int], ratios: tuple[float, float], imageIndex: int, imagePath: str) -> None:
        
        # self.image: Tensor = image
        self.image: ndarray = cv2.imread(imagePath)
        self.label: ndarray = label
        self.shape: tuple[float, float] = shape
        self.segmentSize: tuple[int, int] = segmentSize
        self.ratios: tuple[float, float] = ratios
        self.imageIndex: int = imageIndex
        self.imagePath: str = imagePath

        self.augment: bool = False

        maskPath: str = os.path.join(segmentPath, segmentName + '.mask')
        labelsMaskPath: str = os.path.join(segmentPath, segmentName + '.labels')

        ts: tuple[int, int] = (self.shape[1] - self.segmentSize[1] + 1, self.shape[0] - self.segmentSize[0] + 1)

        self.mask: Tensor = torch.from_numpy(np.fromfile(maskPath, dtype=np.bool_)).reshape(ts)
        self.labelsMask: Tensor = torch.from_numpy(np.fromfile(labelsMaskPath, dtype=np.bool_)).reshape((self.label.shape[0], *ts))
        
        self.iMask: Tensor = torch.logical_and(self.mask[None].cuda(), self.labelsMask.cuda()).cpu()

        self.iNz: list[Tensor] = [self.iMask[i].nonzero() for i in range(self.iMask.shape[0])]
        # self.iNz = [torch.cat((torch.zeros((iNz.shape[0], 1), dtype=int32) + self.imageIndex, iNz.flip(1)), dim=1) for iNz in self.iNz if iNz.shape[0] > 0]
        self.iNz = [iNz.flip(1) for iNz in self.iNz if iNz.shape[0] > 0]

        self.segmentCount: int = len(self.iNz)

        self.count: int = -1

    def __len__(self):
        return self.segmentCount
    
    def __iter__(self):
        self.count = -1
        return self
    
    def nextRandomSegment(self, index: int):
        self.count += 1
        
        iNz: Tensor = self.iNz[self.count % self.segmentCount]
        iNz = iNz[np.random.randint(0, iNz.shape[0])]

        xmin = iNz[0].item()
        ymin = iNz[1].item()

        label = self.label.copy()

        # recenter and rescale
        label[:, 1] = self.ratios[0] * (label[:, 1] - xmin / self.shape[0])
        label[:, 2] = self.ratios[1] * (label[:, 2] - ymin / self.shape[1])
        label[:, 3] = self.ratios[0] * (label[:, 3] - xmin / self.shape[0])
        label[:, 4] = self.ratios[1] * (label[:, 4] - ymin / self.shape[1])

        label = np.concatenate((np.zeros((label.shape[0], 1), dtype=np.float32) + index, label), 1)

        # filter labels that end outside of the segment
        label = label[label[:, 4] > 0]
        label = label[label[:, 5] > 0]
        label = label[label[:, 2] < 1]
        label = label[label[:, 3] < 1]

        # constrain label to the segment
        label[label[:, 2] < 0, 2] = 0
        label[label[:, 3] < 0, 3] = 0
        label[label[:, 4] > 1, 4] = 1
        label[label[:, 5] > 1, 5] = 1

        # from xyxy to xywh
        lc: ndarray = label.copy()
        label[:, 2] = (lc[:, 2] + lc[:, 4]) / 2
        label[:, 3] = (lc[:, 3] + lc[:, 5]) / 2
        label[:, 4] = lc[:, 4] - lc[:, 2]
        label[:, 5] = lc[:, 5] - lc[:, 3]
        
        return self.image.copy(), label, self.imagePath, xmin, ymin

def numpy_to_scaled_tensor(imagesNp: ndarray, device: torch.device, shape: tuple[int, int] = (416, 416)) -> Tensor:
    with torch.no_grad():
        images: Tensor = torch.from_numpy(imagesNp).cuda().float()[None].permute(0, 3, 1, 2)
        images = F.interpolate(images, size=shape, mode='bilinear').to(device)
    return images

class LegoData:
    ClassMap = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    def __init__(self, path: str, shape: tuple[float, float] = (416, 416)) -> None:
        self.basePath: str = path
        self.imagePath: str = os.path.join(self.basePath, 'images')
        self.labelPath: str = os.path.join(self.basePath, 'labels')
        self.segmentBasePath: str = os.path.join(self.basePath, 'segments', str(shape))

        self.shape: tuple[float, float] = shape
        self.segmentSize: tuple[int, int] = (-1, -1)
        self.augment: bool = False

        self.labelPaths: list[str] = os.listdir(self.labelPath)
        self.labelNames: list[str] = [f.replace('.txt', '') for f in self.labelPaths]
        self.imagePaths: list[str] = [os.path.join(self.imagePath, f + '.png') for f in self.labelNames]

        self.labelPaths = [os.path.join(self.labelPath, f) for f in self.labelPaths]

        images: list[ndarray] = [cv2.imread(f) for f in self.imagePaths]
        self.images: ndarray = np.ascontiguousarray(np.stack(images).transpose(0, 3, 1, 2), dtype=np.float32) / 255.0
        # with torch.no_grad():
        #     self.images: Tensor = torch.from_numpy(imagesNp).cuda()
        #     self.images = F.interpolate(self.images, scale_factor=(self.shape[1] / self.images.shape[-2], self.shape[0] / self.images.shape[-1])).cpu()

        self.labels: ndarray = []
        for labelPath in self.labelPaths:
            with open(labelPath, 'r') as f:
                lines = f.read().splitlines()
            label: ndarray = np.array([l.split() for l in lines], dtype=np.float32)
            # to xyxy
            lc: ndarray = label.copy()
            label[:, 1] = lc[:, 1] - lc[:, 3] / 2
            label[:, 3] = lc[:, 1] + lc[:, 3] / 2
            label[:, 2] = lc[:, 2] - lc[:, 4] / 2
            label[:, 4] = lc[:, 2] + lc[:, 4] / 2
            self.labels.append(label)

        self.sampleCount: int = len(self.labels)

        self.segments: list[LegoSegment] = []
        self.batchSize: int = 1
        self.samplesPerEpoch: int = self.sampleCount
        self.segmentCount: int = len(self.segments)
        self.batchCount: int = 0

        dummyImagePath: str = './coco'
        with open(os.path.join(dummyImagePath, 'trainvalno5k.txt'), 'r') as f:
            self.dummyImagePaths: list[str] = [os.path.join(dummyImagePath, x.strip()) for x in f.readlines()]
        self.dummyImageCount: int = len(self.dummyImagePaths)

    def __len__(self):
        return self.batchCount

    def rebatch(self, augment: bool, batchSize: int, batchCount: int = 0, segmentSize: tuple[int, int] = None) -> None:
        self.augment: bool = augment

        if segmentSize is None:
            segmentSize = (int(self.shape[0]), int(self.shape[1]))
        
        if segmentSize[0] != self.segmentSize[0] or segmentSize[1] != self.segmentSize[1]:
            self.segmentSize = segmentSize
            segmentPath: str = os.path.join(self.segmentBasePath, str(self.segmentSize))
            ratios: tuple[float, float] = ((1.0 * self.shape[0] / self.segmentSize[0], 1.0 * self.shape[1] / self.segmentSize[1]))
            self.segments = [LegoSegment(segmentPath, self.labelNames[i], self.images[i], self.labels[i], self.shape, self.segmentSize, ratios, i, self.imagePaths[i]) for i in range(len(self.labelNames))]
            self.segments = [ls for ls in self.segments if ls.segmentCount > 0]
            self.segmentCount = len(self.segments)
        
        for ls in self.segments:
            ls.augment = self.augment

        self.batchSize = batchSize

        if batchCount < 1:
            self.batchCount = math.floor(self.segmentCount / self.batchSize)
            self.samplesPerEpoch = self.segmentCount
        else:
            self.batchCount = batchCount
            self.samplesPerEpoch = self.batchSize * self.batchCount
    
    def __iter__(self):
        self.count = -1
        # self.shuffled: ndarray = np.random.permutation(self.samplesPerEpoch) % self.sampleCount
        return self

    def __next__(self) -> tuple[Tensor, Tensor, list[str]]:
        self.count += 1
        if self.count == self.batchCount:
            raise StopIteration
        
        ni: int = self.count * self.batchSize
        nf: int = min((self.count + 1) * self.batchSize, self.samplesPerEpoch)
        batchSize = nf - ni

        batchList: list[tuple(Tensor, ndarray, str)] = [self.segments[(ni + i) % self.segmentCount].nextRandomSegment(i) for i in range(batchSize)]

        imageList: list[ndarray] = []
        labelList: list[ndarray] = []
        imagePaths: list[str] = []
        xmins: Tensor = torch.zeros((len(batchList),), dtype=torch.int32)
        ymins: Tensor = torch.zeros((len(batchList),), dtype=torch.int32)
        for x, (i, l, p, xmin, ymin) in enumerate(batchList):
            xmins[x] = xmin 
            ymins[x] = ymin

            if self.augment:
                augment_hsv = True
                if augment_hsv:
                    # SV augmentation by 50%
                    fraction = 0.50
                    img_hsv = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)
                    S = img_hsv[:, :, 1].astype(np.float32)
                    # V = img_hsv[:, :, 2].astype(np.float32)

                    a = (random.random() * 2 - 1) * fraction + 1
                    S *= a
                    if a > 1:
                        np.clip(S, a_min=0, a_max=255, out=S)

                    a = (random.random() * 2 - 1) * fraction + 1
                    # V *= a
                    # if a > 1:
                    #     np.clip(V, a_min=0, a_max=255, out=V)

                    img_hsv[:, :, 1] = S.astype(np.uint8)
                    # img_hsv[:, :, 2] = V.astype(np.uint8)
                    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=i)
            
            imageList.append(i)
            labelList.append(l)
            imagePaths.append(p)
        
        with torch.no_grad():
            images = torch.from_numpy(np.stack(imageList)).cuda().float().flip(3).permute(0, 3, 1, 2) / 255.0
            images = F.interpolate(images, scale_factor=(self.shape[1] / images.shape[-2], self.shape[0] / images.shape[-1]))
            images = torch.cat([images[x, :, ymins[x]:ymins[x]+self.segmentSize[1], xmins[x]:xmins[x]+self.segmentSize[0]].unsqueeze(0) for x in range(images.shape[0])]).permute(0, 2, 3, 1).cpu().numpy()
            # images = torch.cat([images[x, :, ymins[x]:ymins[x]+self.segmentSize[1], xmins[x]:xmins[x]+self.segmentSize[0]].unsqueeze(0) for x in range(images.shape[0])]).cpu()

        for i in range(len(labelList)):
            if self.augment:
                af = False
                if af:
                    images[i], labelList[i], M = random_affine(images[i], labelList[i], degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.90, 1.10))

                # random left-right flip
                lrFlip = True
                if lrFlip & (random.random() > 0.50):
                    images[i] = np.fliplr(images[i])
                    labelList[i][:, 2] = 1 - labelList[i][:, 2]

                # random up-down flip
                udFlip = True
                if udFlip & (random.random() > 0.50):
                    images[i] = np.flipud(images[i])
                    labelList[i][:, 3] = 1 - labelList[i][:, 3]

        with torch.no_grad():
            images = torch.from_numpy(images).cuda().permute(0, 3, 1, 2).cpu()
        
        labels: ndarray = np.concatenate(labelList, 0)
        # labels[:, 1] = LegoData.ClassMap[labels[:, 1].astype(np.int32)]

        return images, torch.from_numpy(labels), imagePaths

def plot_one_box(x, img, label=None, color=None):
    tl = round(0.002 * max(img.shape[0:2])) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

class Darknet3Data:
    def __init__(self, dirs: list[str], shape: tuple[float, float] = (416, 416)) -> None:
        self.augment = False
        self.images = torch.zeros((0, 3, shape[1], shape[0]))
        self.imagePaths = []
        self.batchSize = 0
        self.batchCount = 0
        self.samplesPerEpoch = self.batchCount * self.batchSize
        self.shape = shape
        self.labels: list[ndarray] = []
        for i, d in enumerate(dirs):
            datapath = os.path.join(d, 'train')
            fileNames = [os.path.join(datapath, f) for f in os.listdir(datapath) if f[0] != '_']
            fileNames = sorted(fileNames)
            if os.path.splitext(fileNames[0])[1] == '.txt':
                fileNames = [(fileNames[i], fileNames[i + 1]) for i in range(0, len(fileNames), 2)]
            else:
                fileNames = [(fileNames[i + 1], fileNames[i]) for i in range(0, len(fileNames), 2)]
            images = []
            for labelPath, imagePath in fileNames:
                with open(labelPath, 'r') as f:
                    lines = f.read().splitlines()
                label: ndarray = np.array([l.split() for l in lines], dtype=np.float32)
                # to xyxy
                # lc: ndarray = label.copy()
                label[:, 0] += i
                # label[:, 1] = lc[:, 1] - lc[:, 3] / 2
                # label[:, 3] = lc[:, 1] + lc[:, 3] / 2
                # label[:, 2] = lc[:, 2] - lc[:, 4] / 2
                # label[:, 4] = lc[:, 2] + lc[:, 4] / 2
                if label.shape[0] > 0:
                    self.labels.append(label)
                    with torch.no_grad():
                        image: Tensor = torch.from_numpy(cv2.imread(imagePath)).float().unsqueeze(0).cuda().permute(0, 3, 1, 2) / 255.0
                        images.append(F.interpolate(image, scale_factor=(shape[1] / image.shape[-2], shape[0] / image.shape[-1])).cpu())
                    self.imagePaths.append(imagePath)
            self.images = torch.cat((self.images, *images), dim=0)
        self.sampleCount = self.images.shape[0]

    def rebatch(self, augment: bool, batchSize: int, batchCount: int = 0) -> None:
        self.augment: bool = augment
        self.batchSize = batchSize
        if batchCount > 0:
            self.batchCount = batchCount
        else:
            self.batchCount = self.sampleCount
        self.samplesPerEpoch = self.batchSize * self.batchCount

    def __iter__(self):
        self.count = -1
        self.shuffled: ndarray = np.random.permutation(self.samplesPerEpoch) % self.sampleCount
        return self
        
    def __len__(self):
        return self.batchCount

    def __next__(self) -> tuple[Tensor, Tensor, list[str]]:
        self.count += 1
        if self.count == self.images.shape[0]:
            raise StopIteration

        ni: int = self.batchSize * self.count
        nf: int = self.batchSize * (self.count + 1)

        # batchList: list[tuple(ndarray, ndarray, str)] = [(cv2.imread(self.imagePaths[self.count]), self.labels[self.count], self.imagePaths[self.count])]

        labelList: list[ndarray] = [self.labels[i] for i in self.shuffled[ni:nf]]
        imagePaths: list[str] = [self.imagePaths[i] for i in self.shuffled[ni:nf]]
        with torch.no_grad():
            images: Tensor = self.images[self.shuffled[ni:nf]].clone().cuda()
            for i in range(images.shape[0]):
                l = labelList[i]
                l = np.concatenate((np.zeros((l.shape[0], 1), dtype=np.float32) + i, l), axis=1)
                if self.augment:
                    augmentHsv = True
                    if augmentHsv:
                        fraction = 0.50
                        fw = 2 * fraction
                        images[i:i+1] = TF.adjust_saturation(images[i:i+1], fw * random.random() - fraction + 1)
                        images[i:i+1] = TF.adjust_contrast(images[i:i+1], fw * random.random() - fraction + 1)
                        images[i:i+1] = TF.adjust_hue(images[i:i+1], fraction * (random.random() - 0.5))
                        images[i:i+1] = TF.adjust_gamma(images[i:i+1], fw * random.random() - fraction + 1)
                        images[i:i+1] = TF.adjust_brightness(images[i:i+1], fw * random.random() - fraction + 1)
                        images[i:i+1] = TF.adjust_sharpness(images[i:i+1], fw * random.random() - fraction + 1)
                lrFlip = True
                if lrFlip and (random.random() > 0.50):
                    images[i:i+1] = TF.hflip(images[i:i+1])
                    l[:, 2] = 1 - l[:, 2]
                udFlip = True
                if udFlip and (random.random() > 0.50):
                    images[i:i+1] = TF.vflip(images[i:i+1])
                    l[:, 3] = 1 - l[:, 3]
                labelList[i] = l
            images = images.cpu()
        
        labels = np.concatenate(labelList, 0)
        lc = labels.copy()

        return images, torch.from_numpy(labels), imagePaths




if __name__ == '__main__':
    # ld = Darknet3Data(['screwdriver'])
    # cs = CocoSubset(416, classList=['person', 'cat'])
    ld = LegoData('Lego', shape=(416, 416))
    ld.rebatch(True, 1)
    # print(ld.batchCount)
    windowname = 'test'
    cv2.namedWindow(windowname)
    for i, (timgs, labels, path) in enumerate(ld):
        imgs = (timgs.permute(0, 2, 3, 1).numpy()[:, :, :, ::-1] * 255).astype(np.uint8)
        print(imgs.shape, labels.shape, path)
        print(labels)
        for i in range(imgs.shape[0]):
            img = imgs[i].copy()
            for j, cls, x, y, w, h in labels:
                if int(j) == i:
                    # Add bbox to the image
                    plot_one_box([(img.shape[1] - 1) * (x - w/2), (img.shape[0] - 1) * (y - h/2), (img.shape[1] - 1) * (x + w/2), (img.shape[0] - 1) * (y + h/2)], img)
            cv2.imshow(windowname, img)
            if cv2.waitKey(0) == 27:
                exit(0)
    cv2.destroyAllWindows()
