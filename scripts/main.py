import cv2
import numpy as np
import matplotlib.pyplot as plt
from clearance_estimation import HeightEstimation

height_estimated_hist = []
height_estimated_moving_avg = []
height_estimated_moving_avg_avg = []
bbox_params_hist = np.load('inputs/frames/bbox.npy')
#frame_count = len(bbox_params_hist) 
frame_count = 68

for i in range(frame_count):
    filename = str(i+1) + str('.jpg')
    filepath = str('inputs/frames/') + filename
    frame = cv2.imread(filepath)

    scale_percent = 50 
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame_org = cv2.resize(frame, dim)  

    bbox_params = bbox_params_hist[i][1:5] #[x,y,w,h]
    bbox_params[0] = int((bbox_params[0]/400)*960) 
    bbox_params[1] = int((bbox_params[1]/400)*540) 
    bbox_params[2] = int((bbox_params[2]/400)*960) 
    bbox_params[3] = int((bbox_params[3]/400)*540) 
    #bbox_params = [444, 170, 311, 160] 

    frame_bbox = frame_org.copy()
    frame_bbox = cv2.rectangle(frame_bbox, (bbox_params[0], bbox_params[1]), (bbox_params[0]+bbox_params[2], bbox_params[1]+bbox_params[3]), (255,0,0), 2)
    frame_est = HeightEstimation(frame_org, frame_bbox, bbox_params, np.mean(height_estimated_moving_avg))
    height_overlay_image = frame_est.get_height()
    height_estimated = frame_est.height
    height_estimated_hist.append(height_estimated)
    height_estimated_moving_avg.append(np.mean(height_estimated_hist))
    height_estimated_moving_avg_avg.append(np.mean(height_estimated_moving_avg))

    cv2.imwrite('outputs/' + filename.rsplit(".", 1)[0] + '-overlay.jpg', height_overlay_image)
    print('Clearance height is: ', round(height_estimated,2))

x = np.arange(1,69) 
plt.title("Height Estimate by Aurecle", fontsize=20) 
plt.xlabel("Frame", fontsize=15)
plt.ylabel("Estimated Height $[ft]$", fontsize=15) 
plt.plot(x, height_estimated_hist, label='Raw estimate') 
plt.plot(x, height_estimated_moving_avg, label='Filtered estimate') 
plt.legend(fontsize=13)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()