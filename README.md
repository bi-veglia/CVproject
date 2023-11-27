# CVproject
This repository contains a method to detect objects, identifying labels and reading the related QR code in images/videos/stream. We have utilized a You Only Look Once version 8 (YOLO v8) pretrained model to detect the objects and labes and pyzbar to read the QR code. The method has the advantages of high accuracy and real-time performance, thanks to YOLO architecture. 
The example contained in the read_labels_from_video.ipynb notebook processes the input video and produces a video with added bounding-boxes containing the QR code readout.
