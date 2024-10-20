from transformers import (
    SegformerFeatureExtractor, 
    SegformerForSemanticSegmentation
)
from .config import VIS_LABEL_MAP as LABEL_COLORS_LIST
from .utils import (
    draw_segmentation_map, 
    image_overlay,
    predict
)

import argparse
import cv2
import os
import glob
import time
# from extract_point_locate import find_area_between_points
# import extract_features
from .extract_features import find_area_between_points_optimized

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    help='path to the input image directory',
    default='./inputs/images'
)
parser.add_argument(
    '--device',
    default='cpu:0',
    help='compute device, cpu or cuda'
)
parser.add_argument(
    '--imgsz', 
    default=[400, 300],
    type=int,
    nargs='+',
    help='width, height'
)
parser.add_argument(
    '--model',
    default='./model_iou'
)
args = parser.parse_args()

out_dir = './outputs/images'
os.makedirs(out_dir, exist_ok=True)

extractor = SegformerFeatureExtractor()
model = SegformerForSemanticSegmentation.from_pretrained(args.model)
model.to(args.device).eval()

image_paths = glob.glob(os.path.join(args.input, '*'))
for image_path in image_paths:
    image = cv2.imread(image_path)
    if args.imgsz is not None:
        image = cv2.resize(image, (args.imgsz[0], args.imgsz[1]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    start_time = time.time()
    # Get labels.
    labels = predict(model, extractor, image, args.device)
    # print(labels)
    # toado = find_coordinates(labels)
    # print(find_area_between_points(labels))

    A, B, C, D,areaAB, areaAC = find_area_between_points_optimized(labels)
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    print(fps)

    # Get segmentation map.
    seg_map = draw_segmentation_map(
        labels.cpu(), LABEL_COLORS_LIST
    )
    outputs = image_overlay(image, seg_map)
    cv2.imshow('Image', outputs)
    cv2.waitKey(1)
    # time.sleep(20)
    
    
    # Save path.
    image_name = image_path.split(os.path.sep)[-1]
    save_path = os.path.join(
        out_dir, '_'+image_name
    )
    cv2.imwrite(save_path, outputs)