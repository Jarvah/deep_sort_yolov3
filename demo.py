#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from timeit import time
import warnings
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np

'''
from PIL import Image
from yolo import YOLO
'''
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
warnings.filterwarnings('ignore')

import _thread
tracker_on=False
line_on=False

id_colors=[]
for i in range(32):
	id_colors.append(list(np.random.random(size=3) * 256))

def distance_to_line(x,y, point_A, point_B):
    d=(x-point_A[0])*(point_B[1]-point_A[1])-(y-point_A[1])*(point_B[0]-point_A[0])
    return d


def main():

    from PIL import Image
    from yolo import YOLO

    from deep_sort import preprocessing
    from deep_sort import nn_matching
    from deep_sort.detection import Detection
    from deep_sort.tracker import Tracker
    from tools import generate_detections as gdet
    #warnings.filterwarnings('ignore')
   # Definition of the parameters

    def remove_large_bbox(box):
        if box[3] > frame.shape[1] * 0.6 and box[4] > frame.shape[0] * 0.6:
            return True
        else:
            return False
    yolo=YOLO()

    if tracker_on:
        max_cosine_distance = 0.3
        nn_budget = None
        nms_max_overlap = 1.0
        model_filename = 'model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric, max_iou_distance=0.7, max_age=30, n_init=3)


    #video_path='/home/waiyang/crowd_counting/faster_rcnn/video/TownCentreXVID.avi'
    video_path='/home/waiyang/crowd_counting/Dataset/Panasonic_360degree_Network_Camera.mp4'


    writeVideo_flag = True 
    
    video_capture = cv2.VideoCapture(video_path)

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_fps = video_capture.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter('/home/waiyang/crowd_counting/Dataset/360camera_out.avi', fourcc, video_fps, (w, h))
        #list_file = open('detection.txt', 'w')
        frame_index = -1 
        
    fps = 0.0
    avg_time=0

    # define boundaries
    if line_on:
        point_A = (0, int(h/2))
        point_B = (int(w), int(h/2))
    #(568,15),(1845,240)
    #(0,296),(1757,1009)

        out_l = []
        in_l = []
        out_n=0
        in_n=0

    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

       # image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        origin_boxes = yolo.detect_image(image)
        print(len(origin_boxes))
        boxs = origin_boxes
        #for b in origin_boxes:
        #    if remove_large_bbox(b):
        #        continue
        #    else:
        #        boxs.append(b)
        if tracker_on:
            features = encoder(frame,boxs)
        
        # score to 1.0 here).
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
            #call the tracker
            tracker.predict()
            tracker.update(detections)

        # Run non-maxima suppression.
        #boxes = np.array([d.tlwh for d in detections])
        #scores = np.array([d.confidence for d in detections])
        #indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        #detections = [detections[i] for i in indices]

        
        #for track in tracker.tracks:
        #    if not track.is_confirmed() or track.time_since_update > 1:
        #        continue
        #    bbox = track.to_tlbr()
            #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            #cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

            for det in detections:
                bbox = det.to_tlbr()
                cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 1)
                cv2.putText(frame, str(det.index), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, id_colors[int(det.index) % 32], 1)

            #draw boundaries
            if line_on:
                cv2.line(frame, point_A, point_B, (0, 255, 255), 2)


            #count people in and out features
                for track in tracker.tracks:
                    if not track.is_confirmed():
                        continue
                    if len(track.history)<2:
                        continue
                    if track.count == True:
                        continue
                    d=[]
                    for pos in track.history:
                        d.append(distance_to_line(pos[0],pos[1],point_A, point_B))

                    if d[0]<0 and d[1]>0:
                        out_n+=1
                        out_l.append(track.track_id)
                        track.count=True

                    if d[0]>0 and d[1]<0:
                        in_n+=1
                        in_l.append(track.track_id)
                        track.count=True
                cv2.putText(frame, 'out: '+str(out_n), (20,20), cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1.0, color=(0, 128 ,255), thickness=1)
                cv2.putText(frame, 'in: ' + str(in_n), (20, 30), cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1.0, color=(0, 128, 255), thickness=1)


        else:
            for bbox in boxs:
                cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])),(255,0,0), 1)
                #cv2.putText(frame, str(det.index), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, id_colors[int(det.index) % 32], 1)


        cv2.imshow('', frame)
        
        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            #list_file.write(str(frame_index)+' ')
            #if len(boxs) != 0:
                #for i in range(0,len(boxs)):
                    #list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            #list_file.write('\n')

        t2=time.time()-t1
        if avg_time == 0:
            avg_time = t2
        else:
            avg_time = (avg_time + t2) / 2
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))
        print(avg_time)
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        #list_file.close()
    cv2.destroyAllWindows()

def dummy_loop():
    while True:
        a=1



main()

