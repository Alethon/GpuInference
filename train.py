import os
import argparse
import time
import shutil

import test  # Import test.py to get mAP after each epoch
from utils.datasets import *
from utils.utils import *

from Darknet3Data import *
from Darknet.Darknet3 import Darknet3

def train(cfg, data_cfg, resume=False, epochs=270, batchSize=16, multi_scale=False):
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    weights = os.path.join('.', 'weights')
    if not os.path.isdir(weights):
        os.makedirs(weights)

    latest = os.path.join(weights, 'latest.pt')
    best = os.path.join(weights, 'best.pt')

    data = LegoData('Lego', batchSize, 2)

    dataset_info = readDatasetInfo(os.path.join('cfg', 'obj.data'))
    model = Darknet3(dataset_info['classes'])

    lr0: float = 0.001  # initial learning rate
    lr: float = lr0
    startEpoch: int = 0
    bestLoss = float('inf')
    if resume:
        # checkpoint = torch.load(latest, map_location='cuda')
        checkpoint = torch.load(latest)
        model.load_state_dict(checkpoint['model'])
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=lr0, momentum=.9)
        startEpoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            bestLoss = checkpoint['bestLoss']
        del checkpoint  # current, saved
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr0, momentum=.9)
    
    t0 = time.time()
    # n_burnin = min(round(len(data) / 5 + 1), 1000)  # burn-in batches
    for epoch in range(startEpoch, epochs):
        model = model.to(device=device)
        model.train()

        print(('\n%8s%12s' + '%10s' * 7) % ('Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))

        # Update scheduler (manual)
        if epoch % 25 == 0:
            lr = lr0 / 10
        else:
            lr = lr0
        for x in optimizer.param_groups:
            x['lr'] = lr

        ui = -1
        rloss = defaultdict(float)
        for i, (images, targets, _) in enumerate(data):
            targets = targets.to(device)
            nT = targets.shape[0]
            if nT == 0:  # if no targets continue
                continue

            # SGD burn-in
            # if (epoch == 0) and (i <= n_burnin):
            #     lr = lr0 * (i / n_burnin) ** 4
            #     for x in optimizer.param_groups:
            #         x['lr'] = lr

            prediction = model(images.to(device))
            targetList = build_targets(model, targets, prediction)
            loss, loss_dict = compute_loss(prediction, targetList)
            loss.backward()

            # Running epoch-means of tracked metrics
            ui += 1
            for key, val in loss_dict.items():
                rloss[key] = (rloss[key] * ui + val) / (ui + 1)

            s = ('%8s%12s' + '%10.3g' * 7) % (
                '%g/%g' % (epoch, epochs - 1),
                '%g/%g' % (i, len(data) - 1),
                rloss['xy'], rloss['wh'], rloss['conf'],
                rloss['cls'], rloss['total'],
                nT, time.time() - t0)
            t0 = time.time()

            print(s)

        # Update best loss
        if rloss['total'] < bestLoss:
            bestLoss = rloss['total']

        # Save training results
        save = True
        if save:
            model = model.cpu()
            optimizer = optimizer.cpu()

            # Save latest checkpoint
            checkpoint = {'epoch': epoch,
                          'bestLoss': bestLoss,
                          'model': model.state_dict(),
                          'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, latest)

            # Save best checkpoint
            if bestLoss == rloss['total']:
                shutil.copy(latest, best)
                # os.system('cp ' + latest + ' ' + best)

            # Save backup weights every 5 epochs (optional)
            if (epoch > 0) and (epoch % 5 == 0):
                shutil.copy(latest, weights + 'backup{}.pt'.format(epoch))
                # os.system('cp ' + latest + ' ' + weights + 'backup{}.pt'.format(epoch))

        # Calculate mAP
        with torch.no_grad():
            P, R, mAP = test.test(cfg, data_cfg, weights=latest, batchSize=batchSize, model=model)

        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '%11.3g' * 3 % (P, R, mAP) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=270, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=12, help='size of each image batch')
    parser.add_argument('--accumulate', type=int, default=1, help='accumulate gradient x batches before optimizing')
    parser.add_argument('--cfg', type=str, default='../cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='../cfg/coco.data', help='coco.data file path')
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='pixels')
    parser.add_argument('--resume', action='store_true', help='resume training flag', default=1)
    opt = parser.parse_args()
    print(opt, end='\n\n')

    init_seeds()

    train(
        opt.cfg,
        opt.data_cfg,
        img_size=opt.img_size,
        resume=opt.resume,
        epochs=opt.epochs,
        batchSize=opt.batchSize,
        accumulate=opt.accumulate,
        multi_scale=opt.multi_scale,
    )
