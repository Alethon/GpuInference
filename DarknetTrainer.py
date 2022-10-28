import os
from time import time
import shutil
from collections import defaultdict

import torch
from torch import Tensor

from Darknet3Data import *
from Darknet.Darknet3 import *

# intersection over union area
def whIou(box1: Tensor, box2: Tensor) -> Tensor:
    # Returns the IoU of wh1 to wh2. wh1 is 2, wh2 is nx2
    box2 = box2.t()
    # w, h = box1
    w1, h1 = box1[0], box1[1]
    w2, h2 = box2[0], box2[1]
    # Intersection area
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    # Union Area
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area  # iou

def buildTargets(yoloLayers: list[YoloLayer], targets: Tensor, predictions: list[Tensor]) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor], list[tuple[Tensor, Tensor, Tensor, Tensor]]]:
    # anchors = closest_anchor(model, targets)  # [layer, anchor, i, j]
    txy: list[Tensor] = []
    twh: list[Tensor] = []
    tcls: list[Tensor] = []
    tconf: list[Tensor] = []
    indices: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
    for i, layer in enumerate(yoloLayers):
        nGx: Tensor = layer.nGx  # grid size
        nGy: Tensor = layer.nGy  # grid size
        anchorVector: Tensor = layer.anchorVector
        # iou of targets-anchors
        gwh = torch.cat(((targets[:, 4].cuda() * nGx).cuda(), (targets[:, 5].cuda() * nGy).cuda()), 1)
        # gwh = targets[:, 4:6] * nG
        iou = [whIou(x, gwh) for x in anchorVector]
        iou, a = torch.stack(iou, 0).max(0)  # best iou and anchor
        # reject below threshold ious (OPTIONAL)
        reject = True
        if reject:
            j = iou > 0.01
            t, a, gwh = targets[j], a[j], gwh[j]
        else:
            t = targets
        # Indices
        b, c = t[:, 0:2].long().t()  # target image, class
        gxy = torch.cat((t[:, 2] * nGx, t[:, 3] * nGy), 1)
        # gxy = t[:, 2:4] * nG
        gi, gj = gxy.long().t()  # grid_i, grid_j
        indices.append((b, a, gj, gi))
        # XY coordinates
        txy.append(gxy - gxy.floor())
        # Width and height
        twh.append(torch.log(gwh / anchorVector[a]))  # yolo method
        # twh.append(torch.sqrt(gwh / anchor_vec[a]) / 2)  # power method
        # Class
        tcls.append(c)
        # Conf
        tci = torch.zeros_like(predictions[i][..., 0])
        tci[b, a, gj, gi] = 1  # conf
        tconf.append(tci)
    return txy, twh, tcls, tconf, indices

def computeLoss(p: list[Tensor], targets: tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor], list[tuple[Tensor, Tensor, Tensor, Tensor]]]):  # predictions, targets
    FT = torch.cuda.FloatTensor if p[0].is_cuda else torch.FloatTensor
    loss, lxy, lwh, lcls, lconf = FT([0]), FT([0]), FT([0]), FT([0]), FT([0])
    txy, twh, tcls, tconf, indices = targets
    MSE = nn.MSELoss()
    CE = nn.CrossEntropyLoss()
    BCE = nn.BCEWithLogitsLoss()
    # Compute losses
    # gp = [x.numel() for x in tconf]  # grid points
    for i, pi0 in enumerate(p):  # layer i predictions, i
        b, a, gj, gi = indices[i]  # image, anchor, gridx, gridy
        # Compute losses
        k = 1  # nT / bs
        if len(b) > 0:
            pi = pi0[b, a, gj, gi]  # predictions closest to anchors
            lxy += k * MSE(torch.sigmoid(pi[..., 0:2]), txy[i])  # xy
            lwh += k * MSE(pi[..., 2:4], twh[i])  # wh
            lcls += (k / 4) * CE(pi[..., 5:], tcls[i])
        # pos_weight = FT([gp[i] / min(gp) * 4.])
        # BCE = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        lconf += (k * 64) * BCE(pi0[..., 4], tconf[i])
    loss = lxy + lwh + lconf + lcls
    # Add to dictionary
    d = defaultdict(float)
    losses = [loss.item(), lxy.item(), lwh.item(), lconf.item(), lcls.item()]
    for name, x in zip(['total', 'xy', 'wh', 'conf', 'cls'], losses):
        d[name] = x
    return loss, d

def xywh2xyxy(x: Tensor) -> Tensor:
    y = torch.zeros_like(x)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y

def bbox_iou(box1, box2, x1y1x2y2=True):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

    return inter_area / union_area  # iou

def nonMaxSuppression(prediction: Tensor, confThresh: float = 0.5, nmsThresh: float = 0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    output = [None for _ in range(len(prediction))]
    for image_i, pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        class_prob, class_pred = torch.max(F.softmax(pred[:, 5:], 1), 1)
        v = pred[:, 4] > confThresh
        v = v.nonzero().squeeze()
        if len(v.shape) == 0:
            v = v.unsqueeze(0)

        pred = pred[v]
        class_prob = class_prob[v]
        class_pred = class_pred[v]

        # If none are remaining => process next image
        nP = pred.shape[0]
        if not nP:
            continue

        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_prob, class_pred)
        detections = torch.cat((pred[:, :5], class_prob.float().unsqueeze(1), class_pred.float().unsqueeze(1)), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique().to(prediction.device)

        nms_style = 'OR'  # 'OR' (default), 'AND', 'MERGE' (experimental)
        for c in unique_labels:
            # Get the detections with class c
            dc = detections[detections[:, -1] == c]
            # Sort the detections by maximum object confidence
            _, conf_sort_index = torch.sort(dc[:, 4] * dc[:, 5], descending=True)
            dc = dc[conf_sort_index]

            # Non-maximum suppression
            det_max = []
            ind = list(range(len(dc)))
            if nms_style == 'OR':  # default
                while len(ind):
                    j = ind[0]
                    det_max.append(dc[j:j + 1])  # save highest conf detection
                    reject = bbox_iou(dc[j], dc[ind]) > nmsThresh
                    [ind.pop(i) for i in reversed(reject.nonzero())]

            elif nms_style == 'AND':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < nmsThresh]  # remove ious > threshold

            elif nms_style == 'MERGE':  # weighted mixture box
                while len(dc) > 0:
                    iou = bbox_iou(dc[0], dc[0:])  # iou with other boxes
                    i = iou > nmsThresh

                    weights = dc[i, 4:5] * dc[i, 5:6]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[iou < nmsThresh]

            if len(det_max) > 0:
                det_max = torch.cat(det_max)
                # Add max detections to outputs
                output[image_i] = det_max if output[image_i] is None else torch.cat((output[image_i], det_max))

    return output

def computeAp(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def apPerClass(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(np.concatenate((pred_cls, target_cls), 0))

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = sum(target_cls == c)  # Number of ground truth objects
        n_p = sum(i)  # Number of predicted objects

        if (n_p == 0) and (n_gt == 0):
            continue
        elif (n_p == 0) or (n_gt == 0):
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = np.cumsum(1 - tp[i])
            tpc = np.cumsum(tp[i])

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(tpc[-1] / (n_gt + 1e-16))

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(tpc[-1] / (tpc[-1] + fpc[-1]))

            # AP from recall-precision curve
            ap.append(computeAp(recall_curve, precision_curve))

    return np.array(ap), unique_classes.astype('int32'), np.array(r), np.array(p)

class Darknet3Trainer:
    def __init__(self, datasetInfoPath: str) -> None:
        info: dict[str, any] = readDatasetInfo(datasetInfoPath)
        
        # basic
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classCount: int = info['classes']
        self.weightsPath: str = WEIGHTS
        self.latestWeightsPath: str = os.path.join(self.weightsPath, 'latest.pt')
        self.bestWeightsPath: str = os.path.join(self.weightsPath, 'best.pt')
        if not os.path.isdir(self.weightsPath):
            os.makedirs(self.weightsPath)
        
        # data
        self.data: LegoData = LegoData(info['dataPath'])
        
        # model
        if info['useTiny']:
            self.model: DarknetTiny3 = DarknetTiny3(self.classCount).to(self.device)
            self.yolos: list[YoloLayer] = [self.model.yolo1, self.model.yolo2]
        else:
            self.model: Darknet3 = Darknet3(self.classCount).to(self.device)
            self.yolos: list[YoloLayer] = [self.model.yolo1, self.model.yolo2, self.model.yolo3]

        # training variables
        self.epoch: int = 0
        self.lr: float = 0.001
        self.bestLoss: float = float('inf')
        self.optimizer: torch.optim.Optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
    
    def saveCheckpoint(self, checkPointPath: str) -> None:
        self.model = self.model.cpu()
        torch.save({
            'epoch': self.epoch,
            'bestLoss': self.bestLoss,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, checkPointPath)
        self.model = self.model.to(self.device)

    def loadCheckpoint(self, checkPointPath: str) -> None:
        self.model = self.model.cpu()
        checkpoint: dict = torch.load(checkPointPath)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, self.model.parameters()), lr=self.lr, momentum=0.9)
        self.epoch = checkpoint['epoch']
        if checkpoint['optimizer'] is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.bestLoss = checkpoint['bestLoss']
        del checkpoint
        self.model = self.model.to(self.device)
    
    def updateLearningRate(self) -> None:
        if self.epoch % 25 == 0:
            self.lr /= 10
        for x in self.optimizer.param_groups:
            x['lr'] = self.lr
    
    def test(self, batchSize: int, confThresh: float = 0.5, nmsThresh: float = 0.4, iouThresh: float = 0.5) -> tuple:
        self.model.eval()
        self.data.rebatch(batchSize, 0)
        mean_mAP, mean_R, mean_P, seen = 0.0, 0.0, 0.0, 0
        print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
        mP, mR, mAPs, TP = [], [], [], []
        AP_accum, AP_accum_count = np.zeros(self.classCount), np.zeros(self.classCount)
        for (images, targets, _) in self.data:
            targets = targets.to(self.device)
            t = time()
            output: Tensor = self.model(images.to(self.device))
            output = nonMaxSuppression(output, confThresh=confThresh, nmsThresh=nmsThresh)

            # Compute average precision for each sample
            for si, detections in enumerate(output):
                labels = targets[targets[:, 0] == si, 1:]
                seen += 1
                if detections is None:
                    # If there are labels but no detections mark as zero AP
                    if len(labels) != 0:
                        mP.append(0), mR.append(0), mAPs.append(0)
                    continue

                # Get detections sorted by decreasing confidence scores
                detections = detections[(-detections[:, 4]).argsort()]

                # If no labels add number of detections as incorrect
                correct = []
                if len(labels) == 0:
                    # correct.extend([0 for _ in range(len(detections))])
                    mP.append(0), mR.append(0), mAPs.append(0)
                    continue
                else:
                    # Extract target boxes as (x1, y1, x2, y2)
                    target_box = xywh2xyxy(labels[:, 1:5]) * img_size
                    target_cls = labels[:, 0]

                    detected = []
                    for *pred_box, conf, cls_conf, cls_pred in detections:
                        # Best iou, index between pred and targets
                        iou, bi = bbox_iou(pred_box, target_box).max(0)

                        # If iou > threshold and class is correct mark as correct
                        if iou > iouThresh and cls_pred == target_cls[bi] and bi not in detected:
                            correct.append(1)
                            detected.append(bi)
                        else:
                            correct.append(0)

                # Compute Average Precision (AP) per class
                AP, AP_class, R, P = apPerClass(tp=np.array(correct),
                                                conf=detections[:, 4].cpu().numpy(),
                                                pred_cls=detections[:, 6].cpu().numpy(),
                                                target_cls=target_cls.cpu().numpy())

                # Accumulate AP per class
                AP_accum_count += np.bincount(AP_class, minlength=self.classCount)
                AP_accum += np.bincount(AP_class, minlength=self.classCount, weights=AP)

                # Compute mean AP across all classes in this image, and append to image list
                mP.append(P.mean())
                mR.append(R.mean())
                mAPs.append(AP.mean())

                # Means of all images
                mean_P = np.mean(mP)
                mean_R = np.mean(mR)
                mean_mAP = np.mean(mAPs)
            # Print image mAP and running mean mAP
            print(('%11s%11s' + '%11.3g' * 4 + 's') % (seen, self.data.sampleCount, mean_P, mean_R, mean_mAP, time.time() - t))
        # Print mAP per class
        print('\nmAP Per Class:')
        for i in range(self.classCount):
            print('%15d: %-.4f' % (i, AP_accum[i] / (AP_accum_count[i])))
        # Return mAP
        return mean_P, mean_R, mean_mAP

    def _trainEpoch(self, save: bool = True) -> None:
        self.epoch += 1
        self.updateLearningRate()
        rloss = defaultdict(float)
        t = time()
        for i, (images, targets, _) in enumerate(self.data):
            # print(images.shape)
            targetCount: int = targets.shape[0]
            prediction: list[Tensor] = self.model(images.to(self.device))
            targetList = buildTargets(self.yolos, targets, prediction)
            loss, loss_dict = computeLoss(prediction, targetList)
            loss.backward()
            for key, val in loss_dict.items():
                rloss[key] = (rloss[key] * i + val) / (i + 1)
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
    
    def trainEpochs(self, epochCount: int, batchSize: int, reuseCount: int, save: bool = True) -> None:
        if epochCount <= 0:
            return
        self.model.train()
        self.data.rebatch(batchSize, reuseCount)
        for _ in range(epochCount):
            self._trainEpoch(save=save)
    
    def trainThenTest(self, epochCount: int, batchSize: int, reuseCount: int, save: bool = True) -> None:
        self.trainEpochs(epochCount, batchSize, reuseCount, save=save)
        with torch.no_grad():
            self.test(batchSize)

if __name__ == '__main__':
    infoPath = os.path.join('.', 'cfg', 'obj.data')
    dt = Darknet3Trainer(infoPath)
    print('initialized')
    dt.trainThenTest(1, 1, 3)
