#!/usr/bin/env python

'''
Contain functions to draw Bird Eye View for region of interest(ROI) and draw bounding boxes according to risk factor
for humans in a frame and draw lines between boxes according to risk factor between two humans. 
'''

__title__           = "main.py"
__Version__         = "1.1"
__author__          = "Amin Nikmanesh"
__email__           = "mohammadamin_nikmanesh@sfu.ca"
__python_version__  = "3.5.2"

# imports
import cv2
import numpy as np

# Function to draw Bird Eye View for region of interest(ROI). Red, Yellow, Green points represents risk to human. 
# Red: High Risk
# Yellow: Low Risk
# Green: No Risk
def bird_eye_view(frame, distances_mat, bottom_points, dis_to_bench_mat, bench_points, ids, scale_w, scale_h, risk_count):
    h = frame.shape[0]
    w = frame.shape[1]

    red = (0, 0, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    white = (200, 200, 200)
    magenta = (255, 0, 255)
    blue = (255, 0, 0)

    blank_image = np.zeros((int(h * scale_h), int(w * scale_w), 3), np.uint8)
    blank_image[:] = white
    warped_pts = []
    r = []
    g = []
    y = []

    for i in range(len(dis_to_bench_mat)):
        blank_image = cv2.line(blank_image, (int(dis_to_bench_mat[i][0][0] * scale_w), int(dis_to_bench_mat[i][0][1] * scale_h)), (int(dis_to_bench_mat[i][1][0] * scale_w), int(dis_to_bench_mat[i][1][1]* scale_h)), blue, 2)
        blank_image = cv2.circle(blank_image, (int(dis_to_bench_mat[i][0][0]  * scale_w), int(dis_to_bench_mat[i][0][1] * scale_h)), 5, blue, 10)

    for i in range(len(distances_mat)):

        if distances_mat[i][2] == 0:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                r.append(distances_mat[i][0])
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                r.append(distances_mat[i][1])

            blank_image = cv2.line(blank_image, (int(distances_mat[i][0][0] * scale_w), int(distances_mat[i][0][1] * scale_h)), (int(distances_mat[i][1][0] * scale_w), int(distances_mat[i][1][1]* scale_h)), red, 2)
            
    for i in range(len(distances_mat)):
                
        if distances_mat[i][2] == 1:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                y.append(distances_mat[i][0])
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                y.append(distances_mat[i][1])
        
            blank_image = cv2.line(blank_image, (int(distances_mat[i][0][0] * scale_w), int(distances_mat[i][0][1] * scale_h)), (int(distances_mat[i][1][0] * scale_w), int(distances_mat[i][1][1]* scale_h)), yellow, 2)
            
    for i in range(len(distances_mat)):
        
        if distances_mat[i][2] == 2:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                g.append(distances_mat[i][0])
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                g.append(distances_mat[i][1])
    
    for i, oid in zip(bottom_points, ids):
        blank_image = cv2.circle(blank_image, (int(i[0]  * scale_w), int(i[1] * scale_h)), 5, green, 10)
        cv2.putText(blank_image, str(oid), (int(i[0]  * scale_w), int(i[1] * scale_h - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, magenta, 1)
    for i in y:
        blank_image = cv2.circle(blank_image, (int(i[0]  * scale_w), int(i[1] * scale_h)), 5, yellow, 10)
    #    cv2.putText(blank_image, str(oid), (int(i[0]  * scale_w), int(i[1] * scale_h)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, yellow, 1)
    for i in r:
        blank_image = cv2.circle(blank_image, (int(i[0]  * scale_w), int(i[1] * scale_h)), 5, red, 10)
    #    cv2.putText(blank_image, str(oid), (int(i[0]  * scale_w), int(i[1] * scale_h)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 1)
        
    #pad = np.full((100,blank_image.shape[1],3), [110, 110, 100], dtype=np.uint8)
    #cv2.putText(pad, "-- HIGH RISK : " + str(risk_count[0]) + " people", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    #cv2.putText(pad, "-- LOW RISK : " + str(risk_count[1]) + " people", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    #cv2.putText(pad, "-- SAFE : " + str(risk_count[2]) + " people", (50,  80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    #blank_image = np.vstack((blank_image,pad))   
        
    return blank_image
    
# Function to draw bounding boxes according to risk factor for humans in a frame and draw lines between
# boxes according to risk factor between two humans.
# Red: High Risk
# Yellow: Low Risk
# Green: No Risk 
def social_distancing_view(frame, distances_mat, boxes, bench_bxs_mat, benches, ids, speeds, risk_count, frame_n, frame_time):
    
    red = (0, 0, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    cyan = (255, 255, 0)
    magenta = (255, 0, 255)
    black = (0, 0, 0)
    white = (200, 200, 200)
    blue = (255, 0, 0)
    
    for i in range(len(boxes)):

        x,y,w,h = boxes[i][:]
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),green,2)
        box_str = '{} '.format(str(ids[i]))
        cv2.putText(frame, box_str, (x,y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, red, 2)

    for i in range(len(benches)):

        # x,y,w,h = benches[i][:]
        x, y = benches[i][:]
        # frame = cv2.rectangle(frame,(x,y),(x+w,y+h),blue,2)
        frame = cv2.circle(frame, (int(x), int(y)), 5, blue, 10)
                           
    for i in range(len(bench_bxs_mat)):

        bench = bench_bxs_mat[i][0]
        person = bench_bxs_mat[i][1]
        dist = bench_bxs_mat[i][2]
        
        # x,y,w,h = bench[:]
        x, y = bench[:]
        x1,y1,w1,h1 = person[:]

        frame = cv2.line(frame, (int(x), int(y)), (int(x1+w1/2), int(y1+h1/2)), blue, 2)
        cv2.putText(frame, str(dist), (int((x + x1+w1/2)*0.5), int((y + y1+h1/2)*0.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 3)
        cv2.putText(frame, str(dist), (int((x + x1+w1/2)*0.5), int((y + y1+h1/2)*0.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black, 1) 
      
    
    for i in range(len(distances_mat)):

        per1 = distances_mat[i][0]
        per2 = distances_mat[i][1]
        closeness = distances_mat[i][2]
        dist = distances_mat[i][3]
        
        if closeness == 1:
            x,y,w,h = per1[:]
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),yellow,2)
                
            x1,y1,w1,h1 = per2[:]
            frame = cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),yellow,2)
                
            frame = cv2.line(frame, (int(x+w/2), int(y+h/2)), (int(x1+w1/2), int(y1+h1/2)),yellow, 2)
            cv2.putText(frame, str(dist), (int((x+w/2 + x1+w1/2)*0.5), int((y+h/2 + y1+h1/2)*0.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 3)
            cv2.putText(frame, str(dist), (int((x+w/2 + x1+w1/2)*0.5), int((y+h/2 + y1+h1/2)*0.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black, 1) 
            
    for i in range(len(distances_mat)):

        per1 = distances_mat[i][0]
        per2 = distances_mat[i][1]
        closeness = distances_mat[i][2]
        dist = distances_mat[i][3]
        
        if closeness == 0:
            x,y,w,h = per1[:]
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),red,2)
                
            x1,y1,w1,h1 = per2[:]
            frame = cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),red,2)
                
            frame = cv2.line(frame, (int(x+w/2), int(y+h/2)), (int(x1+w1/2), int(y1+h1/2)),red, 2)
            cv2.putText(frame, str(dist), (int((x+w/2 + x1+w1/2)*0.5), int((y+h/2 + y1+h1/2)*0.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 3)
            cv2.putText(frame, str(dist), (int((x+w/2 + x1+w1/2)*0.5), int((y+h/2 + y1+h1/2)*0.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black, 1) 
            
    pad = np.full((90,frame.shape[1],3), [110, 110, 100], dtype=np.uint8)
    cv2.putText(pad, "Frame Number: {}".format(frame_n), (50, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 250, 250), 2)
    cv2.putText(pad, "Frame Time: {} ms".format(frame_time), (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 250, 250), 2)
    #cv2.putText(pad, "Bounding box shows the level of risk to the person.", (50, 110),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 0), 2)
    #cv2.putText(pad, "-- HIGH RISK : " + str(risk_count[0]) + " people", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    #cv2.putText(pad, "-- LOW RISK : " + str(risk_count[1]) + " people", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    #cv2.putText(pad, "-- SAFE : " + str(risk_count[2]) + " people", (50,  180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    frame = np.vstack((frame,pad))
            
    return frame

