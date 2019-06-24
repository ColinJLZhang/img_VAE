#!/usr/env/python 3.5
# -*-coding:utf-8-*-

import os
import numpy as np
import cv2
from PIL import Image
import pylab as plt

SCALE = 0.2

#video_path = r"E:\postgraduate\渔机所\烟台东方海洋出差\实验视频及图片\GOPR6437.MP4"
video_path = r"C:\迅雷下载\[阳光电影www.ygdy8.com].冰川时代5：星际碰撞.BD.720p.中英双字幕.mkv"
img_path = r"E:\python_practice_code\RVAE\dev\imgdata"


def frame2img(video, sample_rate, start=32, end=600):
    """
    get frame from video by sample_rate from start to end
    start\\end: second
    """
    frame_rate = 30   
    start = start * frame_rate
    end = end * frame_rate

    cap = cv2.VideoCapture(video)
    cap.set(1, start)  # 从32s开始正常的鱼类运动图像
    while cap.isOpened:
        ret, frame = cap.read()
        # 获取当前的帧数
        count = int(cap.get(1))
        if count % sample_rate == 0:
            print(count)
            frame = cv2.resize(frame, (int(frame.shape[1] * SCALE), int(frame.shape[0] * SCALE)), interpolation=cv2.INTER_CUBIC)
            frame.astype(int)
            cv2.imwrite(os.path.join(img_path,'f%d.png' % count), frame)
            cv2.imshow("video", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        if cap.get(1) >= end:
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    frame2img(video_path, 5)
