import numpy as np
import csv
import os
import pandas as pd
import torch
import cv2
import scipy.io

df    = pd.read_csv('../Mind_Wandering_Detection_Data_Korea/fold_ids.csv')
participants_fail = {}

part_id = df['participant_id'].values.tolist()
uuid = df['uuid'].values.tolist()
mw_label = df['mw_label'].values.tolist()
start_frame = df['frame'].values.tolist()
participants = []

vidpath = '../Mind_Wandering_Detection_Data_Korea/data/'
outpath = '../Mind_Wandering_Detection_Data_Korea/data_separate_videos/'


for idx in range(len(part_id)):
    if part_id[idx] not in participants:
        participants.append(part_id[idx])
        vidfile = cv2.VideoCapture(vidpath+part_id[idx]+'/'+'recording.mp4')
        if (vidfile.isOpened() == False):
            print("Error opening the video file")
        fps = vidfile.get(cv2.CAP_PROP_FPS)
        frame_count = vidfile.get(cv2.CAP_PROP_FRAME_COUNT)

        frames = []
        while(vidfile.isOpened()):
            ret, frame = vidfile.read()
            if ret == True:
                #frame = np.reshape(frame, (3, 480, 640))
                frames.append(frame)
            else:
                break
        vidfile.release()
        if not os.path.exists(outpath+part_id[idx]):
              os.makedirs(outpath+part_id[idx])
    currect_video_frames = frames[start_frame[idx]-300:start_frame[idx]]
    out = cv2.VideoWriter(outpath+part_id[idx]+'/'+uuid[idx]+".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    for fr in currect_video_frames:
        out.write(fr) # frame is a numpy.ndarray with shape (1280, 720, 3)
    out.release()



        