#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################


import cv2
import argparse
import numpy as np
import os
# ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image', required=True,
#                 help = 'path to input image')
# args = ap.parse_args()
CONFIG  = './yolov3_resource/yolov3.cfg'
CLASSES = './yolov3_resource/yolov3.txt'
WEIGHTS = './yolov3_resource/yolov3.weights'

# Download weights's file
import wget
if os.path.isfile(WEIGHTS)==False:
    print("Downloading YOLOv3 weights...")
    wget.download('https://pjreddie.com/media/files/yolov3.weights', out="./yolov3_resource")

net = cv2.dnn.readNet(WEIGHTS, CONFIG)

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


# def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
#
#     label = str(classes[class_id])
#
#     color = COLORS[class_id]
#
#     cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
#
#     cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
def detectMultiScale(image, scale):
    Width = image.shape[1]
    Height = image.shape[0]
    # scale = 0.00392
    classes = None

    with open(CLASSES, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))


    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    person_rects = []
    for i in indices:
        i = i[0]
        if str(classes[class_ids[i]]) == 'person':
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            if(x<0):
                x=0
            if (y<0):
                y = 0
            person_rects.append([round(x),round(y),round(w), round(h)])
    # print(person_rects)
    return person_rects
        # draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

# cv2.imshow("object detection", image)

# cv2.imwrite("object-detection.jpg", image)
