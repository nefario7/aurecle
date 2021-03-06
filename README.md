# AuReClE - Augemented Reality Cleareance Estimation
## Summary
There have been various incidents of overheight vehicles getting stuck under bridges, underpasses, etc around the world. To state a few such incidents, an Air India plane got stuck  under a bridge when it was being transported by road, a semi tractor-trailer got stuck under an underpass in Milwaukee, among others. Thus, these incidents happen frequently and primarily arise due to insufficient reconnaissance or something as trivial as human error. Because, it is evident that despite human eyes being extremely capable, humans cannot estimate distances and make estimations accurately.  

In the US, it has become abundantly clear over the past few years that infrastructure systems, including transportation networks, have become increasingly poor in terms of safety and capacity. Just between 2015 and 2018, there have been more than 130 overheight vehicle-related crashes in just the state of Washington. Furthermore, apart from just the vehicle getting damaged, there is significant damage to the infrastructure as well. Almost 80% of all damages to bridges are due to overheight vehicle crashes. Moreover, places with a significantly high number of bridges and underpasses are more prone to such incidents. According to the Texas Department of Transportation, Texas has the most bridges and overpasses, approximately 51,000 of them, and each incident costs an average of $180,000 USD.  

To avoid such collisions, there exist some passive solutions such as warning lights and road signages. Unfortunately, most of these systems are only limited to some specific areas and are not ubiquitous. Due to the sheer number of possible locations where such collisions could happen, it’s not viable to install such warning systems everywhere. At the same time, it would incur a high cost of maintaining them. Moreover, with the growth of e-commerce and other global markets, the supply chains are operating at their peak capacity. As trucks and their loads continue to grow bigger, hence increasing the risk of damage to older bridges and structures.

Therefore instead of relying on existing and passive solutions, we propose an active warning system for logistic vehicles that can work anywhere, bridges, underpasses, tunnels, etc. Using a perception system, our system will be able to estimate the clearance in the upcoming structure, and our system would give a pre-emptive go-no-go warning through a heads-up display to the driver as per the driven vehicles’ dimensions. Our system, the Augmented Reality Clearance Estimation, AuReClE, can be retrofitted into existing logistic vehicles with a perception system and a display for convenience.  

AuReClE would utilize a Computer Vision-based model and an AR Pipeline along with a heads-up display to estimate clearances and inform the user of the critical warnings. With real-time perception data, our image segmentation stack extracts possible obstacles and a mask to estimate the obstacle dimensions and size. Further, we approximate the distance between the vehicle and the obstacle to scaling the obstacle dimensions. On comparing it with the pre-defined vehicle dimensions we give a warning to the user on the heads-up display. Now, integrate AuReClE with any vehicle’s heads-up display and make way for a safe, efficient, and cost-effective logistics experience.  

## Installation
```
pip install -r requirements.txt
```
## Segmentation
1. Run main.py --process "segmentation" --directory "processing/input_to_main/" argument which runs segmentation and thresholding pipeline and stores it in processing/input_to_main/   


## Height Estimation

1. Run video_to_frames.py to convert input video into frames. 
Input video filename: “input/input_video.mp4” (default)
frames saved to dir: “processing/input_to_main/” (default)

2. Run main.py --process "heightestimation" argument which calls clearance_estimation.py to take input frames & .npy file (bounding box parameters) from the dir “input_to_main/“.
Default frame saved to dir: “processing/input_to_main/” 
Output frames with clearance markings will be saved to dir “processing/output_from_main/”

3. Run video_compiler.py to create output video from the frames in dir “processing/output_from_main“. Output video with extension .avi will be saved to dir “output/”

## Repository Structure
```
├── input                           # Input video
├── misc                            # Miscellanious test scripts
├── output                          # Output video
├── processing                      # Intermediate processing files
│   ├── input_to_main
│   └── output_from_main
├── clearance_estimation.py         # Height estimation class                     
├── processing.py                   # Frame processing functions
├── slic.py                         # SLIC segmentation class
├── utils.py                        # Utility functions for segmentation
├── video_compiler.py               # Convert frames to video
├── video_to_frames.py              # Convert video to frames
└── main.py     
```    


