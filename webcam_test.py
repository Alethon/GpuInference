import argparse
import shutil
import time
from pathlib import Path

from utils.datasets import *
from utils.utils import *
from utils.parse_config import *

from Darknet.Darknet3 import Darknet3

def detect(
        cfg,
        weights,
        images,
        output='output',  # output folder
        img_size=416,
        save_txt=False,
        conf_thres=0.3,
        nms_thres=0.45
):
    device = torch_utils.select_device()
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    model = Darknet3(2)

    model.load_state_dict(torch.load(weights, map_location='cpu')['model'])

    model.to(device).eval()

    # Set Dataloader
    save_images = False
    dataloader = LoadWebcam()

    # Get classes and colors
    classes = load_classes(parse_data_cfg('./cfg/coco.data')['names'])
    colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]

    for i, (path, img, im0) in enumerate(dataloader):
        t = time.time()
        print('webcam frame %g: ' % (i + 1), end='')
        save_path = str(Path(output) / Path(path).name)

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        pred = model(img)
        pred = pred[pred[:, :, 4] > conf_thres]  # remove boxes < threshold

        if len(pred) > 0:
            # Run NMS on predictions
            detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0]

            # Rescale boxes from 416 to true image size
            scale_coords(img_size, detections[:, :4], im0.shape).round()

            # Print results to screen
            unique_classes = detections[:, -1].cpu().unique()
            for c in unique_classes:
                n = (detections[:, -1].cpu() == c).sum()
                print('%g %ss' % (n, classes[int(c)]), end=', ')

            # Draw bounding boxes and labels of detections
            for x1, y1, x2, y2, conf, cls_conf, cls in detections:
                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write('%g %g %g %g %g %g\n' %
                                   (x1, y1, x2, y2, cls, cls_conf * conf))

                # Add bbox to the image
                label = '%s %.2f' % (classes[int(cls)], conf)
                plot_one_box([x1, y1, x2, y2], im0, label=label, color=colors[int(cls)])

        dt = time.time() - t
        print('Done. (%.3fs)' % dt)

        if save_images:  # Save generated image with detections
            cv2.imwrite(save_path, im0)

        cv2.imshow(weights, im0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/latest.pt', help='path to weights file')
    parser.add_argument('--images', type=str, default='./data/samples', help='path to images')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(
            opt.cfg,
            opt.weights,
            opt.images,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres
        )
