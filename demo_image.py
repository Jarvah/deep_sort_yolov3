from __future__ import division, print_function, absolute_import

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from timeit import time
import warnings
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import glob



from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

warnings.filterwarnings('ignore')

import _thread

id_colors = []
for i in range(32):
    id_colors.append(list(np.random.random(size=3) * 256))

yolo = YOLO()
nms_max_overlap=1.0

ROI_COORDS={
    'IMM':[(4,32),(273,100),(639,233),(639,478),(268,479),(2,264)],
    'jCube':[(9,298),(629,88),(633,472),(8,468)],
    'WestGate':[(6,199),(634,40),(633,472),(8,468)],
    'towncenter':[(11,442),(1048,1),(1918,2),(1918,1075),(3,1074)]
}


def distance_to_line(x, y, point_A, point_B):
    d = (x - point_A[0]) * (point_B[1] - point_A[1]) - (y - point_A[1]) * (point_B[0] - point_A[0])
    return d


def main():
    #img_dir = '/home/waiyang/crowd_counting/faster_rcnn/video/images'
    #output_dir = '/home/waiyang/crowd_counting/keras-yolo3/yolov3_output/towncenter'
    img_dir = '/home/waiyang/crowd_counting/Dataset/test_image_20190527/IMM'
    output_dir = '/home/waiyang/crowd_counting/keras-yolo3/yolov3_output/IMM1'
    img_dir='/home/waiyang/crowd_counting/Dataset/360camera/image'
    output_dir='/home/waiyang/crowd_counting/Dataset/360camera/output_img'
    #for Infolder in glob.glob(os.path.join(img_dir, '*')):
    # video_name = output_dir+'/'+Infolder.replace(img_dir,'').replace('/','')+'.avi'
    # print(video_name)

    # video = cv2.VideoWriter(video_name, 0, 1, (640, 480))
        #for inInfolder in glob.glob(os.path.join(Infolder, '*')):
    for inImg in glob.glob(os.path.join(img_dir,'*')):

            #for inImg in glob.glob(os.path.join(Infolder, '*jpg')):
                print(inImg)
                try:
                #image = Image.open(inImg)
                    image=cv2.imread(inImg)
            # img = np.array(image)
            # coordinateStore1 = CoordinateStore(img)
                except:
                    print('Open Error! Try again!')
                    continue
                else:
                    result,num=algorithm(image, None)

                    #cv2.imwrite(
                    #    output_dir + Infolder.replace(img_dir, '') + inImg[:-4].replace(inInfolder, '') + "_output.jpg",
                    #    result)
                    cv2.imwrite(output_dir+inImg[:-4].replace(img_dir,'')+"_out.jpg",result)
    yolo.close_session()

def algorithm(frame,ROI_coords):
        def remove_large_bbox(box):
            if box[2] > frame.shape[1] * 0.7 and box[3] > frame.shape[0] * 0.7:
                return True
            else:
                return False

        t1 = time.time()

        # image = Image.fromarray(frame)
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        origin_boxes = yolo.detect_image(image)


        boxes = []
        for b in origin_boxes:
            if remove_large_bbox(b):
                continue
            else:
                boxes.append(b)

        #roi applied
        if ROI_coords!=None:
            # draw points
            #for point in ROI_coords:
            #    cv2.circle(frame,point,1,(0,255,255),5)
            ctr=np.array(ROI_coords).reshape((-1,1,2)).astype(np.int32)
            cv2.drawContours(frame, [ctr], 0, (0, 255, 255), 2)

        # Call the tracker


        # for track in tracker.tracks:
        #    if not track.is_confirmed() or track.time_since_update > 1:
        #        continue
        #    bbox = track.to_tlbr()
        # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
        # cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
        num=0
        roi_boxes=[]
        for det in boxes:

            bbox = [int(det[0]),int(det[1]),int(det[0]+det[2]),int(det[1]+det[3])]

            #whether inside ROI region
            if ROI_coords!=None:
                dist = cv2.pointPolygonTest(ctr, (int((bbox[0] +bbox[2] ) / 2), int((bbox[1] + bbox[3]) / 2)), True)
                if dist>=0:
                    num+=1

                    roi_boxes.append(bbox)
                    #masking
                    frame[bbox[1]:int(det[1]+det[3]/2),bbox[0]:bbox[2]] = cv2.GaussianBlur(
                        frame[bbox[1]:int(det[1]+det[3]/2),bbox[0]:bbox[2]], (45, 45),0)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 1)

            #cv2.putText(frame, str(det.index), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200,
             #           id_colors[int(det.index) % 32], 1)
            else:
                #frame[bbox[1]:int(det[1] + det[3] / 2), bbox[0]:bbox[2]] = cv2.GaussianBlur(
                 #   frame[bbox[1]:int(det[1] + det[3] / 2), bbox[0]:bbox[2]], (45, 45), 0)
                try:
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                    text=det[5]+': '+str(det[4])
                    cv2.putText(frame, text, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_PLAIN, fontScale=1.0,
                                color=(0, 255, 255), thickness=1)
                except Exception as e:
                    print(e)
                num+=1

        #print(len(roi_boxes))
        #print(num)

        text_font = cv2.FONT_HERSHEY_PLAIN

        font_size = 2.0
        rectangle_bgr = (0, 255, 255)

        (text_width, text_height) = cv2.getTextSize(str(num), text_font, fontScale=font_size, thickness=2)[0]
        text_offset_y = 30
        text_offset_x = frame.shape[1] - 50
        box_coords = [(text_offset_x, text_offset_y), (text_offset_x + text_width - 2, text_offset_y - text_height - 2)]
        cv2.rectangle(frame, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
        cv2.putText(frame, str(num), (frame.shape[1] - 50, 30), cv2.FONT_HERSHEY_PLAIN, fontScale=2.0,
                    color=(255, 0, 0), thickness=2)

        #cv2.imshow('result', frame)
        #cv2.waitKey(0)


        t2 = time.time() - t1
        #if avg_time == 0:
        #    avg_time = t2
        #else:
        #    avg_time = (avg_time + t2) / 2
        #fps = (fps + (1. / (time.time() - t1))) / 2
        #print("fps= %f" % (fps))
        #print(t2)
        # Press Q to stop!
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        return frame,num
    #video_capture.release()
    #if writeVideo_flag:
    #    out.release()
    #    list_file.close()
    #cv2.destroyAllWindows()




#main()

def test_one_image():
    img_dir='20150304171403886.jp'

    image = cv2.imread(img_dir)
    #roi_coords=roi_preprocessing2((image))
    result, num = algorithm(image, None)

    # cv2.imwrite(
    #    output_dir + Infolder.replace(img_dir, '') + inImg[:-4].replace(inInfolder, '') + "_output.jpg",
    #    result)
    cv2.imwrite("5cup_out.jpg", result)
    yolo.close_session()


def double_counting_test():
    img1 = cv2.imread('20141211130501117.jpg')
    img2 = cv2.imread('20141211130500597.jpg')



    # before
    # predict bonding box
    b_frame1, b_num1 = algorithm(img1, ROI_coords=None)
    b_frame2, b_num2 = algorithm(img2, ROI_coords=None)
    # merge 2 images
    before = np.concatenate((b_frame1, b_frame2), axis=1)
    # print total counting
    text = "tot count: " + str(b_num1 + b_num2)
    (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, fontScale=2, thickness=2)[0]
    text_offset_y = before.shape[0] - 30
    text_offset_x = before.shape[1] - 700
    box_coords = [(text_offset_x, text_offset_y), (text_offset_x + text_width - 2, text_offset_y - text_height - 2)]
    cv2.rectangle(before, box_coords[0], box_coords[1], (0, 255, 255), cv2.FILLED)
    cv2.putText(before, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_PLAIN, fontScale=2.0,
                color=(255, 0, 0), thickness=2)
    cv2.imwrite('before.jpg', before)
    cv2.imshow('before', before)
    cv2.waitKey(0)


    #after
    img1 = cv2.imread('20141211130501117.jpg')
    img2 = cv2.imread('20141211130500597.jpg')

    # apply roi to img1
    roi_coord = [(0, 0), (580, 0), (580, 480), (0, 480)]
    a_frame1, a_num1 = algorithm(img1, roi_coord)
    a_frame2, a_num2 = algorithm(img2, ROI_coords=None)
    cv2.line(a_frame1, roi_coord[1], roi_coord[2], (0, 255, 255), 2)
    # merge image
    after = np.concatenate((a_frame1, a_frame2), axis=1)
    # print total count
    text = "tot count: " + str(a_num1 + a_num2)
    (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, fontScale=2, thickness=2)[0]
    text_offset_y = after.shape[0] - 30
    text_offset_x = after.shape[1] - 700
    box_coords = [(text_offset_x, text_offset_y), (text_offset_x + text_width - 2, text_offset_y - text_height - 2)]
    cv2.rectangle(after, box_coords[0], box_coords[1], (0, 255, 255), cv2.FILLED)
    cv2.putText(after, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_PLAIN, fontScale=2.0,
                color=(255, 0, 0), thickness=2)
    cv2.imwrite('after.jpg', after)
    cv2.imshow('after', after)
    cv2.waitKey(0)
    yolo.close_session()

def roi_preprocessing(img):
    class roi_coordstore:
        def __init__(self,frame):
            self.points=[]
            self.img=frame
            self.first=False
            self.done = False  # Flag signalling we're done
            self.current = (0, 0)  # Current position, so we can draw the line-in-progress

        def draw_contour(self, event, x, y, flags, param):
            global pt1_x,pt1_y
            if event == cv2.EVENT_LBUTTONDOWN:
                # Left click means adding a point at current position to the list of points
                print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))

                self.points.append((x, y))
            elif event == cv2.EVENT_MOUSEMOVE:
                # We want to be able to draw the line-in-progress, so update current mouse position
                self.current = (x, y)
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Right click means we're done
                print("Completing polygon with %d points." % len(self.points))
                self.done = True


    FINAL_LINE_COLOR = (0, 255, 255)

    roi_coords=roi_coordstore(img)
    cv2.namedWindow('draw ROI')
    cv2.setMouseCallback('draw ROI', roi_coords.draw_contour)
    cv2.imshow('draw ROI', roi_coords.img)
    while (not roi_coords.done):

        if (len(roi_coords.points) > 0):
            cv2.polylines(img, np.array([roi_coords.points]), False, FINAL_LINE_COLOR, 1)
        # And  also show what the current segment would look like

        # Update the window
        cv2.imshow('draw ROI', roi_coords.img)
        if cv2.waitKey(1) & 0xFF == 27:
            roi_coords.done=True
    cv2.destroyAllWindows()
    print(roi_coords.points)
    return roi_coords.points

def roi_preprocessing2(img):
    class roi_coordstore:
        def __init__(self,frame):
            self.points=[]
            self.img=frame
            self.first=False
            self.done = False  # Flag signalling we're done
              # Current position, so we can draw the line-in-progress

        def draw_contour(self, event, x, y, flags, param):
            global pt1_x,pt1_y
            if event == cv2.EVENT_LBUTTONDOWN:
                # Left click means adding a point at current position to the list of points
                print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
                if self.first:
                    cv2.line(self.img, (pt1_x, pt1_y), (x, y), color=(0, 255, 255), thickness=3)
                    pt1_x, pt1_y=x, y
                    self.points.append((x, y))

                else:
                    pt1_x, pt1_y = x, y
                    self.points.append((x, y))
                    self.first = True

            elif event == cv2.EVENT_RBUTTONDOWN:
                # Right click means we're done
                print("Completing polygon with %d points." % len(self.points))
                cv2.line(self.img, (pt1_x, pt1_y), (x, y), color=(0, 255, 255), thickness=3)
                pt1_x, pt1_y = x, y
                self.points.append((x, y))



    FINAL_LINE_COLOR = (0, 255, 255)

    roi_coords=roi_coordstore(img)
    cv2.namedWindow('draw ROI')
    cv2.setMouseCallback('draw ROI', roi_coords.draw_contour)
    cv2.imshow('draw ROI', roi_coords.img)
    while (1):


        cv2.imshow('draw ROI', roi_coords.img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    print(roi_coords.points)
    return roi_coords.points


def split_into_four(frame, enable_360):

        Width, Height = frame.shape[1], frame.shape[0]
        crop_h, crop_w = int(Height / 2), int(Width / 2)

        # Get the 4 cropped frames and put them in an array
        crop1 = frame[0:crop_h, 0:crop_w]
        crop2 = frame[0:crop_h, crop_w:Width]
        crop3 = frame[crop_h:Height, 0:crop_w]
        crop4 = frame[crop_h:Height, crop_w:Width]

        if enable_360:

            crop2 = np.rot90(crop2, k=1, axes=(0, 1))
            new_crop2 = crop2.copy()

            crop3 = np.rot90(crop3, k=2, axes=(0, 1))
            new_crop3 = crop3.copy()
            crop4 = np.rot90(crop4, k=3, axes=(0, 1))
            new_crop4 = crop4.copy()
            crop_array = [crop1, new_crop2, new_crop3, new_crop4]
        else:
            crop_array = [crop1, crop2, crop3, crop4]



        return crop_array

def test_video():
    #video_path = '/home/waiyang/crowd_counting/Dataset/double_para.mp4'
    video_path='/home/waiyang/crowd_counting/Dataset/test.mp4'
    split_into_two=False
    split_into_four=True
    enable_360=True

    writeVideo_flag = True

    video_capture = cv2.VideoCapture(video_path)

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_fps = video_capture.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter('/home/waiyang/crowd_counting/Dataset/test_out.avi', fourcc, video_fps, (w, h))
        # list_file = open('detection.txt', 'w')
        frame_index = -1

    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        #frame=undistort(frame)


        if split_into_two:
            print("crop")
            frame1=np.array(frame[0:frame.shape[0], 0:750])
            frame1=np.rot90(frame1,k=3,axes=(0,1))
            print(frame1.shape)
            #frame1=rotate(frame1,-90)


            frame2=np.array(frame[0:frame.shape[0],750:frame.shape[1]])

            frame2=np.rot90(frame2,k=1,axes=(0,1))
            print(frame2.shape)

            #frame2=rotate(frame2,90)
            img1=frame1.copy()
            img2=frame2.copy()
            #img1=undistort(img1)
            #img2=undistort(img2)
            print("after undistort")
            print(img1.shape)
            print(img2.shape)
            result1, num = algorithm(img1, None)
            result2, num = algorithm(img2, None)
            result1=np.rot90(result1, k=1 ,axes=(0,1))
            result2=np.rot90(result2, k=3 ,axes=(0,1))

        #result1=rotate(result1, 90)
        #result2=rotate(result2, -90)
            print("result")
            print(result1.shape)
            print(result2.shape)
            result=np.concatenate((result1,result2),axis=1)
        elif split_into_four:
            new_crop_array=[]
            crop_array=split_into_four(frame, enable_360)
            for img in crop_array:
                new_crop_array.append(algorithm(img,None))


        else:
            result, num = algorithm(frame, None)

            cv2.imshow('frame', result)

        if writeVideo_flag:
            # save a frame
            out.write(result)
            frame_index = frame_index + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()

    cv2.destroyAllWindows()

def undistort(img):

    DIM = img.shape[:2]
    K = np.array(
        [[781.3524863867165, 0.0, 794.7118000552183], [0.0, 779.5071163774452, 561.3314451453386], [0.0, 0.0, 1.0]])
    D = np.array([[-0.042595202508066574], [0.031307765215775184], [-0.04104704724832258], [0.015343014605793324]])
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


def rotate(image, angle, center=None, scale=1.0):

    (h, w) = image.shape[:2]


    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


#test_video()
test_one_image()
#main()