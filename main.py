#!/usr/bin/env python

'''
Calculates Region of Interest(ROI) by receiving points from mouse event and transform prespective so that
we can have top view of scene or ROI. This top view or bird eye view has the property that points are
distributed uniformally horizontally and vertically(scale for horizontal and vertical direction will be
 different). So for bird eye view points are equally distributed, which was not case for normal view.

YOLO V3 is used to detect humans in frame and by calculating bottom center point of bounding boxe around humans, 
we transform those points to bird eye view. And then calculates risk factor by calculating distance between
points and then drawing birds eye view and drawing bounding boxes and distance lines between boxes on frame.
'''

__title__ = "main.py"
__Version__ = "1.1"
__author__ = "Ali Saberi"
__email__ = "ali.saberi96@gmail.com"
__python_version__ = "3.5.2"

# imports
import os

import cv2
import numpy as np
import time
import argparse
from imutils.video import FPS
import pandas as pd

import torch

# own modules
import utills, plot
from FairMOT.src.lib.opts import opts
from objects import Object  # alit
from pyimagesearch.centroidtracker import CentroidTracker

# FairMOT
from FairMOT.src.lib.tracker.multitracker import JDETracker

confid = 0.3
thresh = 0.5
maxDisappeared = 5  # alit
maxDistance = 30  # alit

mouse_pts = []


def letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


# Function to get points for Region of Interest(ROI) and distance scale. It will take 8 points on first frame using mouse click
# event.First four points will define ROI where we want to moniter social distancing. Also these points should form parallel  
# lines in real world if seen from above(birds eye view). Next 3 points will define 6 feet(unit length) distance in     
# horizontal and vertical direction and those should form parallel lines with ROI. Unit length we can take based on choice.
# Points should pe in pre-defined order - bottom-left, bottom-right, top-right, top-left, point 5 and 6 should form     
# horizontal line and point 5 and 7 should form verticle line. Horizontal and vertical scale will be different. 

# Function will be called on mouse events

def get_mouse_points(event, x, y, flags, param):
    global mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_pts) < 4:
            cv2.circle(image, (x, y), 5, (0, 0, 255), 10)
        elif len(mouse_pts) < 7:
            cv2.circle(image, (x, y), 5, (255, 0, 0), 10)
        else:
            cv2.circle(image, (x, y), 5, (0, 255, 0), 10)

        if len(mouse_pts) >= 1 and len(mouse_pts) <= 3:
            cv2.line(image, (x, y), (mouse_pts[len(mouse_pts) - 1][0], mouse_pts[len(mouse_pts) - 1][1]), (70, 70, 70),
                     2)
            if len(mouse_pts) == 3:
                cv2.line(image, (x, y), (mouse_pts[0][0], mouse_pts[0][1]), (70, 70, 70), 2)

        if "mouse_pts" not in globals():
            mouse_pts = []

        mouse_pts.append((x, y))
        # print("Point detected")
        # print(mouse_pts)


def calculate_social_distancing(opt, output_dir, output_vid, ln1, detection_rate, net):
    use_cuda = opt.gpus != [-1]

    count = 0
    vs = cv2.VideoCapture(opt.video_path)

    # Get video height, width and fps
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps_ = int(vs.get(cv2.CAP_PROP_FPS))
    frame_rate = int(round(fps_))

    # fairMOT
    MOT_tracker = JDETracker(opt, frame_rate=frame_rate)

    if detection_rate < 1:
        detection_rate = frame_rate

    mot_w, mot_h = 1920, 1080

    # Set scale for birds eye view
    # Bird's eye view will only show ROI
    scale_w, scale_h = utills.get_scale(mot_w, mot_h)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    output_movie = cv2.VideoWriter("./output_vid/distancing.avi", fourcc, fps_, (int(width), int(height + 210)))
    bird_movie = cv2.VideoWriter("./output_vid/bird_eye_view.avi", fourcc, fps_,
                                 (int(width * scale_w), int(height * scale_h)))

    # alit    
    # ct = CentroidTracker(maxDisappeared=maxDisappeared, maxDistance=maxDistance)
    # final_tracking_time = 0
    # final_tracking_points = []
    trackableObjects = {}

    fps = FPS().start()  # alit

    points = []
    global image

    datas = []

    oid_counter = 0

    while True:

        print("Frame {} - {} ms".format(count, vs.get(cv2.CAP_PROP_POS_MSEC)))

        (grabbed, frame) = vs.read()
        (H, W) = frame.shape[:2]

        mot_frame0 = cv2.resize(frame, (mot_w, mot_h))

        # Padded resize
        img_size = (1088, 608)
        mot_frame, _, _, _ = letterbox(mot_frame0, height=img_size[1], width=img_size[0])

        # Normalize RGB
        mot_frame = mot_frame[:, :, ::-1].transpose(2, 0, 1)
        mot_frame = np.ascontiguousarray(mot_frame, dtype=np.float32)
        mot_frame /= 255.0

        if not grabbed:
            break

        frame_time = vs.get(cv2.CAP_PROP_POS_MSEC)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # alit

        rects = []  # alit
        ids = []
        boxes1 = []
        speeds = []

        # first frame will be used to draw ROI and horizontal and vertical 180 cm distance(unit length in both directions)
        if count == 0:
            while True:
                image = mot_frame0
                cv2.imshow("image", image)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    cv2.destroyWindow("image")
                    break

            points = mouse_pts

            # Using first 4 points or coordinates for perspective transformation. The region marked by these 4 points are
            # considered ROI. This polygon shaped ROI is then warped into a rectangle which becomes the bird eye view.
            # This bird eye view then has the property property that points are distributed uniformally horizontally and
            # vertically(scale for horizontal and vertical direction will be different). So for bird eye view points are
            # equally distributed, which was not case for normal view.
            src = np.float32(np.array(points[:4]))
            dst = np.float32([[0, mot_h], [mot_w, mot_h], [mot_w, 0], [0, 0]])
            prespective_transform = cv2.getPerspectiveTransform(src, dst)

            # using next 3 points for horizontal and vertical unit length(in this case 180 cm)
            pts = np.float32(np.array([points[4:7]]))
            warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]

            bpts = np.float32(np.array([points[7:]]))
            bench_points = cv2.perspectiveTransform(bpts, prespective_transform)[0]
            bpts = list(bpts[0])

            # since bird eye view has property that all points are equidistant in horizontal and vertical direction.
            # distance_w and distance_h will give us 180 cm distance in both horizontal and vertical directions
            # (how many pixels will be there in 180cm length in horizontal and vertical direction of birds eye view),
            # which we can use to calculate distance between two humans in transformed view or bird eye view
            distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
            distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
            pnts = np.array(points[:4], np.int32)

        cv2.polylines(mot_frame0, [pnts], True, (70, 70, 70), thickness=2)

        ####################################################################################

        if use_cuda:
            blob = torch.from_numpy(mot_frame).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(mot_frame).unsqueeze(0)

        online_targets = MOT_tracker.update(blob, mot_frame0)

        for t in online_targets:
            tlwh = [int(c) for c in t.tlwh]
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                boxes1.append(tlwh)
                ids.append(tid)

                if tid not in trackableObjects.keys():
                    trackableObjects[tid] = Object(tid, None)

        person_points = utills.get_transformed_points(boxes1, prespective_transform)
        # ids = trackableObjects.keys()

        for i, oid in enumerate(ids):
            to = trackableObjects.get(oid, None)
            to.points.append(person_points[i])
            to.times.append(frame_time)

            if count % speed_rate == 0:
                to.cal_speed(distance_w, distance_h, 5, 2)

            speeds.append(to.speed)
            # print('speed', to.speed)

        # if count % detection_rate == 0:
        #
        #     trackableObjects = {}
        #
        #     # YOLO v3
        #     blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        #     net.setInput(blob)
        #     start = time.time()
        #     layerOutputs = net.forward(ln1)
        #
        #     end = time.time()
        #     boxes = []
        #     # benches = []
        #     # benches1 = []
        #     confidences = []
        #     # bench_confs = []
        #     classIDs = []
        #
        #     for output in layerOutputs:
        #         for detection in output:
        #             scores = detection[5:]
        #             classID = np.argmax(scores)
        #             confidence = scores[classID]
        #             # detecting humans in frame
        #             if classID == 0:
        #
        #                 if confidence > confid:
        #                     box = detection[0:4] * np.array([W, H, W, H])
        #                     (centerX, centerY, width, height) = box.astype("int")
        #
        #                     x = int(centerX - (width / 2))
        #                     y = int(centerY - (height / 2))
        #
        #                     boxes.append([x, y, int(width), int(height)])
        #                     confidences.append(float(confidence))
        #                     classIDs.append(classID)
        #
        #             # bench
        #             # if classID == 13:
        #
        #             #     if confidence > confid:
        #
        #             #        bench = detection[0:4] * np.array([W, H, W, H])
        #             #        (centerX, centerY, width, height) = bench.astype("int")
        #
        #             #        x = int(centerX - (width / 2))
        #             #        y = int(centerY - (height / 2))
        #
        #             #        benches.append([x, y, int(width), int(height)])
        #             #        bench_confs.append(float(confidence))
        #             #        classIDs.append(classID)
        #
        #     idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)
        #     # bench_idxs = cv2.dnn.NMSBoxes(benches, bench_confs, confid, thresh)
        #
        #     # font = cv2.FONT_HERSHEY_PLAIN
        #
        #     for i in range(len(boxes)):
        #         if i in idxs:
        #             boxes1.append(boxes[i])
        #             x, y, w, h = [int(v) for v in boxes[i]]  # alit
        #
        #             # alit
        #             tracker = cv2.TrackerCSRT_create()
        #             tracker.init(frame, tuple([int(v) for v in boxes[i]]))
        #
        #             rects.append((x, y, x + w, y + h))
        #
        #             trackableObjects[oid_counter] = Object(oid_counter, tracker)
        #
        #             oid_counter += 1
        #
        #     # for i in range(len(benches)):
        #     #     if i in bench_idxs:
        #     #         benches1.append(benches[i])
        #
        #     # bench_points = utills.get_transformed_points(benches1, prespective_transform)
        #
        #     ct = CentroidTracker(maxDisappeared=maxDisappeared, maxDistance=maxDistance)
        #     person_points = utills.get_transformed_points(boxes1, prespective_transform)
        #     objects = ct.update(person_points)
        #     objects = ct.update(final_tracking_points)
        #
        #     ids = trackableObjects.keys()
        #
        #     for i, oid in enumerate(ids):
        #         to = trackableObjects.get(oid, None)
        #         to.points.append(objects[i])
        #         to.points.append(person_points[i])
        #         to.times.append(final_tracking_time)
        #         to.times.append(frame_time)
        #
        #         if count % speed_rate == 0:
        #             to.cal_speed(distance_w, distance_h, 5, 2)
        #
        #         speeds.append(to.speed)
        #         # print('speed', to.speed)
        #
        #
        # else:
        #
        #     for to in trackableObjects.values():
        #
        #         (success, box) = to.tracker.update(rgb)
        #
        #         box = [int(v) for v in box]
        #         boxes1.append(box)
        #
        #         if success:
        #             (x, y, w, h) = box
        #             rects.append((x, y, x + w, y + h))
        #
        #     person_points = utills.get_transformed_points(boxes1, prespective_transform)
        #
        #     ids = trackableObjects.keys()
        #
        #     for i, oid in enumerate(ids):
        #         to = trackableObjects.get(oid, None)
        #         to.points.append(person_points[i])
        #         to.times.append(frame_time)
        #
        #         if count % speed_rate == 0:
        #             to.cal_speed(distance_w, distance_h, 5, 2)
        #
        #         speeds.append(to.speed)
        #         # print('speed', to.speed)

        fps.update()
        fps.stop()

        # if count % detection_rate == detection_rate - 1:
        #     final_tracking_points = person_points
        #     final_tracking_time = frame_time

        # objects = ct.update(person_points)

        # ids = list(ids)

        distances_mat, bxs_mat = utills.get_distances(boxes1, person_points, ids, distance_w, distance_h)
        dis_to_bench_mat, bench_pts_mat, complete_bench_mat = utills.get_bench_distances(boxes1, person_points, bpts,
                                                                                         bench_points, ids, distance_w,
                                                                                         distance_h)

        for i, oid in enumerate(ids):
            data = {}
            data['Frame No.'] = count
            data['Frame time'] = frame_time
            data['Person ID'] = oid
            data['X'] = person_points[i][0]
            data['Y'] = person_points[i][1]
            data['Speed'] = speeds[i]

            for d in complete_bench_mat:
                if d[4] != oid:
                    continue

                data['Distance to B{}'.format(d[3])] = d[2]

            for d in distances_mat:
                if d[4] != oid:
                    continue

                data['Distance to P{}'.format(d[5])] = d[3]

            datas.append(data)

        # if len(rects) == 0:
        #     count = count + 1
        #     continue

        # alit
        # for (objectID, centroid) in objects.items():
        #     #ids.append(objectID)
        #     to = trackableObjects.get(objectID, None)
        #     if to is None:
        #         to = Object(objectID, centroid)

        #     trackableObjects[objectID] = to

        # Here we will be using bottom center point of bounding box for all boxes and will transform all those
        # bottom center points to bird eye view

        # person_points = utills.get_transformed_points(boxes1, prespective_transform)

        # Here we will calculate distance between transformed points(humans)

        # dis_to_bench_mat, bench_bxs_mat = utills.get_bench_distances(boxes1, person_points, benches1, bench_points, distance_w, distance_h)
        risk_count = utills.get_count(distances_mat)

        # frame1 = np.copy(frame)

        # Draw bird eye view and frame with bouding boxes around humans according to risk factor    
        bird_image = plot.bird_eye_view(mot_frame0, distances_mat, person_points, dis_to_bench_mat, bench_points, ids,
                                        scale_w, scale_h, risk_count)
        # img = plot.social_distancing_view(frame1, bxs_mat, boxes1, bench_bxs_mat, benches1, speeds, risk_count)
        img = plot.social_distancing_view(mot_frame0, bxs_mat, boxes1, bench_pts_mat, bpts, ids, speeds, risk_count, count,
                                          frame_time)

        # Show/write image and videos
        if count != 0:
            output_movie.write(img)
            bird_movie.write(bird_image)

            cv2.imshow('Bird Eye View', bird_image)
            cv2.imshow('Frame', img)  # alit
            cv2.imwrite(output_dir + "frame%d.jpg" % count, img)
            cv2.imwrite(output_dir + "bird_eye_view/frame%d.jpg" % count, bird_image)

        count = count + 1

        keypress = cv2.waitKey(50)

        if keypress & 0xFF == ord('q'):
            break

        if keypress & 0xFF == ord('s'):
            df = pd.DataFrame(datas)
            df.to_excel("social_distancing.xlsx")

            print('==> Data saved')

        keypress = ord('p')

    df = pd.DataFrame(datas)
    df.to_excel("social_distancing.xlsx")

    print('==> Final data saved')

    output_movie.release()
    bird_movie.release()
    vs.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    opt = opts().init()

    model_path = opt.model
    if model_path[len(model_path) - 1] != '/':
        model_path = model_path + '/'

    output_dir = opt.output_dir
    if output_dir[len(output_dir) - 1] != '/':
        output_dir = output_dir + '/'

    output_vid = opt.output_vid
    if output_vid[len(output_vid) - 1] != '/':
        output_vid = output_vid + '/'

    detection_rate = int(opt.detection_rate)  # alit
    speed_rate = int(opt.speed_rate)

    # load Yolov3 weights

    weightsPath = model_path + "yolov3.weights"
    configPath = model_path + "yolov3.cfg"

    net_yl = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net_yl.getLayerNames()
    ln1 = [ln[i[0] - 1] for i in net_yl.getUnconnectedOutLayers()]

    # set mouse callback 

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", get_mouse_points)
    np.random.seed(42)

    calculate_social_distancing(opt, output_dir, output_vid, ln1, detection_rate, net_yl)
