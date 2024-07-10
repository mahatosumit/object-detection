import cv2
from random import randint

# Load YOLOv4-tiny model
dnn = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
model = cv2.dnn_DetectionModel(dnn)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=False)

# Read class names from file
with open('classes.txt') as f:
    classes = f.read().strip().splitlines()

# Open video capture
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

color_map = {}

while True:
    # Capture frame
    ret, frame = capture.read()
    
    # Check if frame is captured successfully
    if not ret or frame is None:
        print("Error: Unable to capture frame.")
        break

    frame = cv2.flip(frame, 1)

    # Object detection
    class_ids, confidences, boxes = model.detect(frame)
    
    for id, confidence, box in zip(class_ids, confidences, boxes):
        x, y, w, h = box
        obj_class = classes[id]

        if obj_class not in color_map:
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
            color_map[obj_class] = color
        else:
            color = color_map[obj_class]

        cv2.putText(frame, f'{obj_class.title()} {format(confidence, ".2f")}', (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Display the frame
    cv2.imshow('Video Capture', frame)

    # Wait for a key press and handle exit or color reset
    key = cv2.waitKey(10)
    
    if key == 27:  # Press 'esc' key to exit
        break

    elif key == 13:  # Press 'enter' key to reset colors
        color_map = {}

# Release video capture and close windows
capture.release()
cv2.destroyAllWindows()
