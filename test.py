import _init_path
import numpy as np
from typing import Optional
from flask import Flask, request, jsonify
import time, requests
from PIL import Image
from io import BytesIO
from pydantic.main import BaseModel
import io
import datetime
import cv2
import base64
import json
import threading
import tool.utils as utils
import config as cfg
from face_models import *
from checkin import Checkin
from tool.sendlog import SendLog
# Python program to save a
# video using OpenCV
import cv2
import datetime
import time
import config as cfg
from face_models import *
from tool.sendlog import SendLog
sendlog = SendLog(cfg.config_sendlog)
face_model = FaceModel(sendlog=sendlog)
# Create an object to read
# from camera
video = cv2.VideoCapture(r"D:\GPBL\VIDEO\CASE 7.mp4")
# We need to check if camera
# is opened previously or not
if (video.isOpened() == False):
    print("Error reading video file")
# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)
# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.
result = cv2.VideoWriter(r"D:\GPBL\VIDEO\CASE 7.avi",
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
while(True):
    ret, frame = video.read()
    if ret == True:
        prev_frame_time = 0
        new_frame_time = 0
        recog_time_st = time.time()
        res, dimg = face_model.face_recog(frame)
        print(res)
        recog_time_en = time.time()
        #print("Recog time", recog_time_en - recog_time_st)
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = round(fps, 2)
        fps = str(fps)
        cv2.putText(dimg, "COWELL", (20,45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)
        cv2.imshow('CAMERA',dimg )
        # Write the frame into the
        # file 'filename.avi'
        result.write(dimg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break
video.release()
result.release()
# Closes all the frames
cv2.destroyAllWindows()
print("The video was successfully saved")