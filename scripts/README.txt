1. Run video_to_frames.py to convert input video into frames. Input video filename: "input_video.mp4". Frames will be saved to dir "input_to_main/"

2. Run main.py which calls clearance_estimation.py to take input frames & a .npy file (bounding box parameters) from the dir "input_to_main/". 
   Output frames with clearance markings will be saved to dir "output_from_main/"

3. Run video_compiler.py to create output video from the frames in dir "output_from_main/". Output video with extension .avi will be saved to dir "final_outputs/"