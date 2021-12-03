import cv2
import os

video_path="input_video.mp4"
input_video = cv2.VideoCapture(video_path)

i=0
while True:
    ret,frame=input_video.read()
    if ret==False:
        input_video.release()
        break
    print(ret)
    i+=1
    name="input_to_main/"+str(i)+".jpg"
    cv2.imwrite(name,frame)

