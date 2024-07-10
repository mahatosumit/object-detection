from ultralytics import YOLO
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load a YOLO model 
model = YOLO('yolov5m.pt')

# access all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
results = model.predict(source="0", show=True)
# results = model.predict(source="folder", show = True)  # Display preds. Accepts all YOLO predict arguments

# # from PIL
# iml = Image.open("bus.jpg")
# results = model.predict(source=im1, save=True) # save plotted images

# # from ndarray
# iml = Image.open("bus.jpg")
# results = model.predict(source=im1, save=True , save_txt=True) # save plotted images


# # from list of PIL/ndarray
# results = model.predict(source=[im1, im2])