import math
import os
import cv2
import numpy as np
import random
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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

class LegoData:
    def __init__(self, path: str) -> None:
        self.basePath: str = path
        self.imagePath: str = os.path.join(self.basePath, 'images')
        self.labelPath: str = os.path.join(self.basePath, 'labels')

        labelPaths: list[str] = os.listdir(self.labelPath)
        imagePaths: list[str] = [os.path.join(self.imagePath, f.replace('.txt', '.png')) for f in labelPaths]
        labelPaths = [os.path.join(self.labelPath, f) for f in labelPaths]

        self.labelPaths: list[str] = []
        self.imagePaths: list[str] = []
        self.labels: list[np.ndarray] = []
        images: list[np.ndarray] = []

        for imagePath, labelPath in zip(imagePaths, labelPaths):
            if os.path.exists(imagePath) and os.path.isfile(imagePath) and os.path.isfile(labelPath):
                self.labelPaths.append(labelPath)
                self.imagePaths.append(imagePath)
                image = cv2.imread(imagePath)
                h, w, _ = image.shape
                with open(labelPath, 'r') as f:
                    lines = f.read().splitlines()
                label: np.ndarray = np.array([l.split() for l in lines], dtype=np.float32)
                label[:, 1] *= w
                label[:, 2] *= h
                label[:, 3] *= w
                label[:, 4] *= h
                if label.shape[0] > 0:
                    self.labels.append(label)
                    images.append(image)
        
        self.images: np.ndarray = np.ascontiguousarray(np.stack(images)[:, :, :, ::-1].transpose(0, 3, 1, 2), dtype=np.float32) / 255.0

        self.sampleCount: int = len(self.labels)

        self.batchSize: int = 1
        self.samplesPerEpoch: int = self.sampleCount
        self.batchCount: int = math.ceil(self.samplesPerEpoch / self.batchSize)

    def __iter__(self):
        self.count = -1
        self.shuffled: np.ndarray = np.random.permutation(self.samplesPerEpoch) % self.sampleCount
        return self

    def rebatch(self, batchSize: int, reuseCount: int) -> None:
        if reuseCount < 0:
            reuseCount = 0
        reuseCount += 1

        self.batchSize = batchSize
        self.samplesPerEpoch = reuseCount * self.sampleCount
        self.batchCount = math.ceil(self.samplesPerEpoch / self.batchSize)
    
    def __next__(self) -> tuple[Tensor, Tensor, list[str]]:
        self.count += 1
        if self.count == self.batchCount:
            raise StopIteration
        
        ni: int = self.count * self.batchSize
        nf: int = min((self.count + 1) * self.batchSize, self.samplesPerEpoch)

        labels: list[np.ndarray] = []
        imagePaths: list[str] = []
        for n in range(ni, nf):
            i = self.shuffled[n]
            label = self.labels[i]
            labels.append(np.concatenate((np.zeros((len(label), 1), dtype=np.float32) + n - ni, label), 1))
            imagePaths.append(self.imagePaths[i])
        
        return torch.from_numpy(self.images[self.shuffled[ni:nf]]), torch.from_numpy(np.concatenate(labels, 0)), imagePaths

if __name__ == '__main__':
    # cs = CocoSubset(416, classList=['person', 'cat'])
    ld = LegoData('Lego')
    ld.rebatch(4, 1)
    for i, (imgs, labels, _) in enumerate(ld):
        print(imgs.shape, labels.shape)
