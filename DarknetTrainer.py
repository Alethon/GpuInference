import os
from time import time
import shutil
from collections import defaultdict

import torch
from torch import Tensor

from utils.utils import *

from DarknetUser import *
from Darknet3Data import *
from Darknet.Darknet3 import *

torch.backends.cudnn.benchmark = True

class Darknet3Trainer(DarknetUser):
    def __init__(self, datasetInfoPath: str) -> None:
        super().__init__(datasetInfoPath)
        
        # data
        self.data: LegoData = LegoData(self.info['dataPath'], shape=(416, 416))
        # self.data: LegoData = Darknet3Data(['screwdriver'])
        
        # self.data: LoadImagesAndLabels = LoadImagesAndLabels(parse_data_cfg('cfg/coco-0-19.data')['train'], 'subsets/0-19/labels', self.classCount, 16, 416, augment=True)
        # self.data: LoadImagesAndLabels = LoadImagesAndLabels('./coco/subsets/0-19/trainvalno5k.txt', 'subsets/0-19/labels', self.classCount, 12, 416, augment=True)

        self.bestWeightsPath: str = os.path.join(self.weightsPath, 'best.pt')
        
        # training variables
        self.epoch: int = 0
        self.lr0: float = 0.001
        self.lr: float = self.lr0
        self.bestLoss: float = float('inf')
        self.optimizer: torch.optim.Optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.nBurnIn = 1000
    
    def updateOptimizer(self) -> None:
        for x in self.optimizer.param_groups:
            x['lr'] = self.lr
    
    def saveCheckpoint(self, checkPointPath: str) -> None:
        self.model = self.model.cpu()
        torch.save({
            'epoch': self.epoch,
            'bestLoss': self.bestLoss,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr': self.lr
        }, checkPointPath)
        self.model = self.model.to(self.device)

    def loadCheckpoint(self, checkPointPath: str) -> dict:
        checkpoint = super().loadCheckpoint(checkPointPath)
        self.lr = checkpoint['lr']
        self.optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, self.model.parameters()), lr=self.lr, momentum=0.9)
        self.epoch = checkpoint['epoch']
        if checkpoint['optimizer'] is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.bestLoss = checkpoint['bestLoss']
        return checkpoint
    
    def updateLearningRate(self) -> None:
        if self.epoch % 250 == 0:
            self.lr /= 10
        self.updateOptimizer()
    
    # def test(self, batchSize: int, confThresh: float = 0.5, nmsThresh: float = 0.4, iouThresh: float = 0.5) -> tuple:
    def test(self, batchSize: int, confThresh: float = 0.3, nmsThresh: float = 0.45, iouThresh: float = 0.5) -> tuple:# largly derived from test.py in https://github.com/ultralytics/yolov3/tree/v3.0 
        self.model.eval()
        mean_mAP, mean_R, mean_P, seen = 0.0, 0.0, 0.0, 0
        self.data.rebatch(False, batchSize)
        print()
        print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
        mP, mR, mAPs = [], [], []
        AP_accum, AP_accum_count = np.zeros(self.classCount), np.zeros(self.classCount)
        for (imgs, targets, _) in self.data:
            targets = targets.to(self.device)
            t = time()
            with torch.no_grad():
                output = self.model(imgs.to(self.device))
            output = non_max_suppression(output, conf_thres=confThresh, nms_thres=nmsThresh)
            for si, detections in enumerate(output):
                labels = targets[targets[:, 0] == si, 1:]
                seen += 1
                if detections is None:
                    if len(labels) != 0:
                        mP.append(0), mR.append(0), mAPs.append(0)
                    continue
                detections = detections[(-detections[:, 4]).argsort()]
                correct = []
                if len(labels) == 0:
                    # correct.extend([0 for _ in range(len(detections))])
                    mP.append(0), mR.append(0), mAPs.append(0)
                    continue
                else:
                    target_box = xywh2xyxy(labels[:, 1:5]) * 416
                    targetClass = labels[:, 0]
                    detected = []
                    for *predictions_box, _, _, cls_predictions in detections:
                        iou, bi = bbox_iou(predictions_box, target_box).max(0)
                        if iou > iouThresh and cls_predictions == targetClass[bi] and bi not in detected:
                            correct.append(1)
                            detected.append(bi)
                        else:
                            correct.append(0)
                AP, AP_class, R, P = ap_per_class(tp=np.array(correct), conf=detections[:, 4].cpu().numpy(), pred_cls=detections[:, 6].cpu().numpy(), target_cls=targetClass.cpu().numpy())
                AP_accum_count += np.bincount(AP_class, minlength=self.classCount)
                AP_accum += np.bincount(AP_class, minlength=self.classCount, weights=AP)
                mP.append(P.mean())
                mR.append(R.mean())
                mAPs.append(AP.mean())
                mean_P = np.mean(mP)
                mean_R = np.mean(mR)
                mean_mAP = np.mean(mAPs)
            print(('%11s%11s' + '%11.3g' * 4 + 's') % (seen, len(self.data), mean_P, mean_R, mean_mAP, time() - t))
        print('\nmAP Per Class:')
        for i in range(self.classCount):
            if AP_accum_count[i]:
                print('%15s: %-.4f' % (self.names[i], AP_accum[i] / (AP_accum_count[i])))
        return mean_P, mean_R, mean_mAP

    def _trainEpoch(self, save: bool = True) -> None: # largly derived from train.py in https://github.com/ultralytics/yolov3/tree/v3.0 
        print(('\n%8s%12s' + '%10s' * 7) % ('Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))
        self.epoch += 1
        self.updateLearningRate()
        rloss = defaultdict(float)
        ui = -1
        t = time()
        for i, (images, targets, _) in enumerate(self.data):
            targetCount: int = targets.shape[0]
            if targetCount == 0:
                continue
            if (self.epoch == 1) and (i <= self.nBurnIn):
                self.lr = self.lr0 * ((i + 1) / self.nBurnIn) ** 4
                self.updateLearningRate()
            targets = targets.to(self.device)
            images = images.to(self.device)
            prediction: list[Tensor] = self.model(images)
            targetList = build_targets(self.yolos, targets, prediction)
            loss, loss_dict = compute_loss(prediction, targetList)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            ui += 1
            for key, val in loss_dict.items():
                rloss[key] = (rloss[key] * ui + val) / (ui + 1)
            s = ('%8s%12s' + '%10.3g' * 7) % (
                '%g' % (self.epoch),
                '%g/%g' % (i, len(self.data) - 1),
                rloss['xy'], rloss['wh'], rloss['conf'],
                rloss['cls'], rloss['total'],
                targetCount, time() - t)
            print(s)
            t = time()
        if rloss['total'] < self.bestLoss:
            self.bestLoss = rloss['total']
        if save:
            self.saveCheckpoint(self.latestWeightsPath)
            if self.bestLoss == rloss['total']:
                shutil.copy(self.latestWeightsPath, self.bestWeightsPath)
            if self.epoch > 0 and self.epoch % 5 == 0:
                shutil.copy(self.latestWeightsPath, os.path.join(self.weightsPath, 'backup{}.pt'.format(self.epoch)))
    
    def trainEpochs(self, epochCount: int, batchSize: int, batchCount: int, save: bool = True) -> None:
        if epochCount <= 0:
            return
        self.model.train()
        self.data.rebatch(True, batchSize, batchCount)
        self.nBurnIn = min(self.nBurnIn, len(self.data))
        for _ in range(epochCount):
            self._trainEpoch(save=save)
    
    def trainThenTest(self, epochCount: int, batchSize: int, batchCount: int, save: bool = True) -> None:
        self.trainEpochs(epochCount, batchSize, batchCount, save=save)
        self.test(batchSize)

if __name__ == '__main__':
    infoPath = os.path.join('.', 'cfg', 'obj.data')
    dt = Darknet3Trainer(infoPath)
    dt.loadCheckpoint(dt.latestWeightsPath)
    # dt.loadCheckpoint(dt.bestWeightsPath)
    dt.test(12)
    while True:
        dt.trainThenTest(1, 12, 200)
