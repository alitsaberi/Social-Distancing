__title__           = "objects.py"
__Version__         = "1.1"
__author__          = "Ali Saberi"
__email__           = "ali.saberi96@gmail.com"

import numpy as np
import cv2
from utills import cal_dis

class Object:

    def __init__(self, oid, tracker):

        self.tracker = tracker
        self.id = oid
        self.points = []
        self.times = []
        self.speed = ''
        self.speeds = []

    def cal_speed(self, distance_w, distance_h, rate, mode=2):

        if mode == 0:
            t = [t1 - t0 for t1, t0 in zip(self.times[1:], self.times)]
            d = [cal_dis(p1, p0, distance_w, distance_h) for p1, p0 in zip(self.points[1:], self.points)]

            v = [int((d_/t_) * 1000)/100 for d_, t_ in zip(d[-1 * rate:], t[-1 * rate:]) if t_ != 0]

            self.speed = int(100 * sum(v) / len(v))/100 if len(v) != 0 else 0 # m/s
        
        elif mode == 1:
            t = self.times[-1] - self.times[-1 * rate] if len(self.times) >= rate else self.times[-1] - self.times[0]
            d = cal_dis(self.points[-1], self.points[-1 * rate], distance_w, distance_h) if len(self.points) >= rate else cal_dis(self.points[-1], self.points[0], distance_w, distance_h)
            self.speed = int(1000 * d / t)/100 if t != 0 else 0 # m/s

        else:
            t = self.times[-1] - self.times[-1 * rate] if len(self.times) >= rate else self.times[-1] - self.times[0]
            d = cal_dis(self.points[-1], self.points[-1 * rate], distance_w, distance_h) if len(self.points) >= rate else cal_dis(self.points[-1], self.points[0], distance_w, distance_h)
            speed = int(1000 * d / t)/100 if t != 0 else 0 # m/s
            self.speeds.append(speed)
            self.speed = int(100 * sum(self.speeds) / len(self.speeds))/100 if len(self.speeds) != 0 else 0 # m/s
