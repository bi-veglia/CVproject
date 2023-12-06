#!/usr/bin/env python
# coding: utf-8

import cv2 as cv



vid = cv.VideoCapture(1) 

frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
size=(frame_width, frame_height)
print(size)

while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
    # Display the resulting frame 
    cv.imshow('frame', frame) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv.destroyAllWindows() 






