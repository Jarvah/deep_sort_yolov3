import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import math
import time
from PIL import Image
import math


def fish_eye_dis(img):
    "fish eye distortion"
    width_in, height_in = img.size
    im_out = Image.new("RGB", (width_in, height_in))
    radius = max(width_in, height_in) / 2
    # assume the fov is 18
    # R = f*theta
    lens = radius * 2 / math.pi
    for i in range(width_in):
        for j in range(height_in):
            # offset to center
            x = i - width_in / 2
            y = j - height_in / 2
            r = math.sqrt(x * x + y * y)
            theta = math.atan(r / radius)
            if theta < 0.00001:
                k = 1
            else:
                k = lens * theta / r

            src_x = x * k
            src_y = y * k
            src_x = src_x + width_in / 2
            src_y = src_y + height_in / 2
            pixel = img.getpixel((src_x, src_y))
            im_out.putpixel((i, j), pixel)

    return im_out




# 鱼眼有效区域截取
def cut(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (_, thresh) = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    x,y,w,h = cv2.boundingRect(cnts)
    r = max(w/ 2, h/ 2)
    # 提取有效区域
    img_valid = img[y:y+h, x:x+w]
    return img_valid, int(r)

# 鱼眼矫正
def undistort(src,r):
    # r： 半径， R: 直径
    R = 2*r
    # Pi: 圆周率
    Pi = np.pi
    # 存储映射结果
    dst = np.zeros((R, R, 3))
    src_h, src_w, _ = src.shape
    # 圆心
    x0, y0 = src_w//2, src_h//2

    for dst_y in range(0, R):

        theta =  Pi - (Pi/R)*dst_y
        temp_theta = math.tan(theta)**2

        for dst_x in range(0, R):
            # 取坐标点 p[i][j]
            # 计算 sita 和 fi

            phi = Pi - (Pi/R)*dst_x
            temp_phi = math.tan(phi)**2

            tempu = r/(temp_phi+ 1 + temp_phi/temp_theta)**0.5
            tempv = r/(temp_theta + 1 + temp_theta/temp_phi)**0.5

            if (phi < Pi/2):
                u = x0 + tempu
            else:
                u = x0 - tempu

            if (theta < Pi/2):
                v = y0 + tempv
            else:
                v = y0 - tempv

            if (u>=0 and v>=0 and u+0.5<src_w and v+0.5<src_h):
                dst[dst_y, dst_x, :] = src[int(v+0.5)][int(u+0.5)]

                # 计算在源图上四个近邻点的位置
                # src_x, src_y = u, v
                # src_x_0 = int(src_x)
                # src_y_0 = int(src_y)
                # src_x_1 = min(src_x_0 + 1, src_w - 1)
                # src_y_1 = min(src_y_0 + 1, src_h - 1)
                #
                # value0 = (src_x_1 - src_x) * src[src_y_0, src_x_0, :] + (src_x - src_x_0) * src[src_y_0, src_x_1, :]
                # value1 = (src_x_1 - src_x) * src[src_y_1, src_x_0, :] + (src_x - src_x_0) * src[src_y_1, src_x_1, :]
                # dst[dst_y, dst_x, :] = ((src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1 + 0.5).astype('uint8')

    return dst

def undistort1():
    t = time.perf_counter()
    frame = cv2.imread('/home/waiyang/crowd_counting/Dataset/360camera/double_para.png')
    cut_img,R = cut(frame)
    result_img = undistort(cut_img,R)
    cv2.imwrite('/home/waiyang/crowd_counting/Dataset/360camera/doublepara_undis.jpg',result_img)
    print(time.perf_counter()-t)

def undistort2():
    input_name = "/home/waiyang/crowd_counting/Dataset/360camera/double_para.png"
    output_name = "/home/waiyang/crowd_counting/Dataset/360camera/doublepara_undis.jpg"
    im = Image.open(input_name)
    img_out = fish_eye_dis(im)
    img_out.save(output_name)

    print( "fish eye distortion completely, save image to %s" % output_name)

undistort1()