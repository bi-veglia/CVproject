#!/usr/bin/env python
# coding: utf-8

import cv2 as cv
from glob import glob
from ultralytics import YOLO
from pyzbar.pyzbar import decode
import math


## if training on custom data is needed ##
#dataset = 'data/label_test2.yaml'
#backbone = YOLO("yolov8s.pt")  # load a pre-trained model (recommended for training)
#results_train = backbone.train(data=dataset, epochs=120,name='label_test2')


rl_model = YOLO('models/best.pt')

videos = glob('data/video/*.mp4')


def qr_reader(qr_crop):
    img=qr_crop
    if img.shape[0]<80:
        img=cv.resize(img, (0,0), fx=7, fy=7)
    detection=decode(img)
    if len(detection)>0:
        text=detection[0].data.decode('utf-8')
        return text
    return None


vid = cv.VideoCapture(1) 
#vid.set(3, 640)
#vid.set(4, 480)
fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('./stream_detected.avi', fourcc, 20.0, (640, 480))
# or if an mp4 file is preferred, use the following lines instead #
# fourcc = cv.VideoWriter_fourcc(*'mp4v')
# out = cv.VideoWriter('./outputs/processed.mp4', fourcc, 20.0, (640, 480))

while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, img = vid.read() 
    frame_out=img
    results = rl_model(img, stream=True)
    for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                #plot box in video
                cv.rectangle(img, (x1,y1),(x2,y2),(255,0,255),5)

                #confidence
                confidence = math.ceil((box.conf[0]*100))/100
                if confidence >0.25:
                    print('Confidence ---->', confidence)
                    crop_label = img[y1:y2, x1:x2]
                    qr_read=qr_reader(crop_label)

                    if qr_read is not None:
                        cv.rectangle(frame_out,(x1,y1),(x2,y2),(0,0,255),5)
                        (text_width, text_height), _ = cv.getTextSize(qr_read, cv.FONT_HERSHEY_SIMPLEX, 2, 6)
                        cv.putText(frame_out,qr_read,(int((x2+x1-text_width)/2), int(y1-text_height)),cv.FONT_HERSHEY_SIMPLEX,2, (0, 255, 0), 5)
                        cv.putText(img,qr_read,(int((x2+x1-text_width)/2), int(y1-text_height)),cv.FONT_HERSHEY_SIMPLEX,2, (0, 255, 0), 5)
    out.write(frame_out)

    cv.imshow('Webcam',img)
    if cv.waitKey(1)== ord('q'):
         break
       

out.release()
vid.release()
cv.destroyAllWindows


