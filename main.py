import os.path
import tkinter
import torch
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import cv2

model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5/runs/train/exp4/weights/last.pt', force_reload=True)

# img = os.path.join('Dataset', 'images', 'sample', '3.jpg')
# results = model(img)
# results.print()
# matplotlib.use('TkAgg')
# plt.imshow(np.squeeze(results.render()))
# plt.show()

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    # detections
    results = model(frame)
    cv2.imshow('YOLO', np.squeeze(results.render()))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# python train.py --img 640 --batch 1 --epochs 3 --data coco128.yaml --weights yolov5s6.pt --cache to train
