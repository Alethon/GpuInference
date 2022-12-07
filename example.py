from CameraUser import *

torch.set_grad_enabled(False)

class Control:
    def __init__(self) -> None:
        self.matchMask = None
        self.matchPred = None
        self.maxTrackingLossTime = 1
        self.matchMaskTime = -1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.camUser = CameraUser('rtsp://192.168.117.59:8554/unicast')
    
    def getNextMask(self, predictions: Tensor, frame: ndarray) -> None:
        t = time()

        if self.matchMask is not None and t - self.matchMaskTime >= self.maxTrackingLossTime:
            self.matchMask = None
        
        if predictions.shape[0] < 1:
            return

        if self.matchMask is None:
            self.matchMaskTime = t
            # get the prediction with the highest confidence
            prediction = predictions[predictions[:, 4].max() == predictions[:, 4]][0]
            self.matchPred = prediction
            # make the new mask
            self.matchMask = torch.zeros((frame.shape[0], frame.shape[1])).to(self.device)
            self.matchMask[i, int(prediction[1]):int(prediction[3]), int(prediction[0]):int(prediction[2])] = prediction[6]
        else:
            predictions = predictions[predictions[:, 6] == self.matchPred[6]]
            if predictions.shape[0] == 0:
                return
            mask = torch.zeros((predictions.shape[0], frame.shape[0], frame.shape[1])).to(self.device)
            for i, (x1, y1, x2, y2, _, _, cls) in enumerate(predictions):
                mask[i, int(y1):int(y2), int(x1):int(x2)] = cls
            intersections = (mask == self.matchMask[None]).float().to(self.device)
            intersections = intersections.sum(2).sum(1)
            if intersections.sum(0) == 0:
                return
            # non-zero intersections
            nzi = intersections > 0
            predictions = predictions[nzi]
            mask = mask[nzi]
            unions = torch.logical_or(self.matchMask[None] > 0, mask > 0).to(self.device).float().sum(2).sum(1)
            iou = torch.divide(intersections, unions)
            bestIndexing = iou == iou.max()
            self.matchMaskTime = t
            self.matchMask = mask[bestIndexing][0]
            self.matchPred = predictions[bestIndexing][0]
    