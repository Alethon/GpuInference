import os
import cv2
import numpy as np
import random
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from utils import *
from ObjectRecognition import Darknet3, DarknetTiny3

CLASS_NAMES = os.path.join('..', 'data', 'coco.names')

COCO = os.path.join('..', 'coco')
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

WEIGHTS = os.path.join('..', 'weights')

def getIntStrLabels(data: list[str]) -> list[tuple[int, str]]:
    return [(int(s[:i]), s[i + 1:]) for i, s in [(l.find(' '), l) for l in data]]

class CocoSubset(Dataset):
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
        self.subsetAlias: str = '-'.join([ogIndexLookup.get(cls) for cls in self.classList])

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

        self._loadData()
    
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
                if i == n:
                    n += 1
                    f5k.write(imagePath)
                else:
                    fno5k.write(imagePath)
    
    def _loadData(self) -> None:
        with open(self.trainvalno5k, 'r') as f:
            data: list[str] = [i.replace('\n', '') for i in f.readlines()]
        for d in data:
            img = torch.from_numpy(cv2.imread(d))
            self.images.append(F.interpolate(, , mode='bilinear'))
            labelPath = d.replace(COCO_IMAGES_UNROOT, self.labelsDirUnroot).replace('.png', '.txt').replace('.jpg', '.txt')
            if os.path.isfile(labelPath):
                with open(labelPath, 'r') as f:
                    labels0 = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                labels0 = labels0[labels0[:, 0] < self.clsCount]
                # Normalized xywh to pixel xyxy format
                labels = labels0.copy()
                labels[:, 1] = ratio * w * (labels0[:, 1] - labels0[:, 3] / 2) + padw
                labels[:, 2] = ratio * h * (labels0[:, 2] - labels0[:, 4] / 2) + padh
                labels[:, 3] = ratio * w * (labels0[:, 1] + labels0[:, 3] / 2) + padw
                labels[:, 4] = ratio * h * (labels0[:, 2] + labels0[:, 4] / 2) + padh
        



class Darknet3User:
    def __init__(self, model: (Darknet3 | DarknetTiny3 | tuple), classList: list[str]) -> None:
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nC: int = len(classList)
        self.classList: list[str] = classList
        if isinstance(model, Darknet3):
            self.model: Darknet3 = model
        elif isinstance(model, DarknetTiny3):
            self.model: DarknetTiny3 = model
        elif isinstance(model, tuple) and model[0] == 'Darknet3':
            self.model: Darknet3 = Darknet3(self.nC)
        elif isinstance(model, tuple) and model[0] == 'DarknetTiny3':
            if len(model) == 3:
                self.model: DarknetTiny3 = DarknetTiny3(self.nC, scale=float(model[2]))
            else:
                self.model: DarknetTiny3 = DarknetTiny3(self.nC)
        else:
            raise TypeError('Bad `model` passed to Darknet3User.')
        
        if isinstance(model, tuple):
            weightfile = os.path.join(WEIGHTS, model[1])
            if os.path.isfile(weightfile):
                self.model.load_state_dict(torch.load(weightfile, map_location='cpu')['model'])
        
        self.model.to(self.device).eval()
        if torch.cuda.is_available():
            self.model.half()
