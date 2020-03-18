import sys
import cv2
import os
from sys import platform
import argparse
import json
import math
import numpy as np
from random import randint


def cal_distance(x1,y1,x2,y2):
    return math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))

def cal_size_bone(bone_struct):

    if bone_struct[1]['x']==0 or bone_struct[1]['y']==0 or bone_struct[8]['x']==0 or bone_struct[8]['y']==0:
        return 0
    return cal_distance(bone_struct[1]['x'],bone_struct[1]['y'],bone_struct[8]['x'],bone_struct[8]['y'])

def draw_bone_line(bone_struct,cv2,image,kp1,kp2):
    if bone_struct[kp1]['x']!=0 and bone_struct[kp1]['y']!=0 and bone_struct[kp2]['x']!=0 and bone_struct[kp2]['y']!=0:
        #print("IM HERE")
        cv2.line(image,(bone_struct[kp1]['x'],bone_struct[kp1]['y']),(bone_struct[kp2]['x'],bone_struct[kp2]['y']),(255,0,0),5)

def draw_bone(bone_struct,cv2,image):
    for i in range(len(bone_struct)):
        # for poseKeypoint in datum.poseKeypoints[0]:
        keypoint = bone_struct[i]
        x_draw = keypoint['x']
        y_draw = keypoint['y']
        # cv2.putText(show_result, '(' + str(int(x_draw)) + ',' + str(int(y_draw)) + ')', (x_draw, y_draw),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(image, (x_draw, y_draw), 5, (0, 0, 0), -1)
    #bone_struct_line=[[0,1],[0,15],[0,16],[1,2],[1,5],[1,8],[2,3],[3,4],[5,6],[6,7],[8,9],[8,12],[9,10],[10,11],[12,13],[13,14],[15,17],[16,18]]
    
    bone_struct_line=[[0,1],[0,15],[0,16],[1,2],[1,5],[1,8],[2,3],[3,4],[5,6],[6,7],[8,9],[8,12],[9,10],[10,11],[12,13],[13,14],[15,17],[16,18],[11,24],[11,22],[22,23],[14,21],[14,19],[19,20]]
    #bone_struct_line=[[0,1],[0,14],[0,15],[1,2],[1,5],[1,8], [1,11],[2,3],[3,4],[5,6],[6,7],[8,9],[9,10],[11,12],[12,13],[14,16],[15,17]]
    for line in bone_struct_line:
        #print(line[0],line[1])
        draw_bone_line(bone_struct,cv2,image,line[0],line[1])

def draw_bone_line_with_color(bone_struct,cv2,image,kp1,kp2,color):
    if bone_struct[kp1]['x']!=0 and bone_struct[kp1]['y']!=0 and bone_struct[kp2]['x']!=0 and bone_struct[kp2]['y']!=0:
        #print("IM HERE")
        cv2.line(image,(bone_struct[kp1]['x'],bone_struct[kp1]['y']),(bone_struct[kp2]['x'],bone_struct[kp2]['y']),color,5)

def draw_bone_with_color(bone_struct,cv2,image,color):

    for i in range(len(bone_struct)):
        # for poseKeypoint in datum.poseKeypoints[0]:
        keypoint = bone_struct[i]
        x_draw = keypoint['x']
        y_draw = keypoint['y']
        # cv2.putText(show_result, '(' + str(int(x_draw)) + ',' + str(int(y_draw)) + ')', (x_draw, y_draw),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(image, (x_draw, y_draw), 5, (0, 0, 0), -1)
    bone_struct_line=[[0,1],[0,15],[0,16],[1,2],[1,5],[1,8],[2,3],[3,4],[5,6],[6,7],[8,9],[8,12],[9,10],[10,11],[12,13],[13,14],[15,17],[16,18],[11,24],[11,22],[22,23],[14,21],[14,19],[19,20]]
    for line in bone_struct_line:
        #print(line[0],line[1])
        draw_bone_line_with_color(bone_struct,cv2,image,line[0],line[1],color)

def box_from_bonestruct(bone_struct):
    min_x=4000
    max_x=0
    min_y=4000
    max_y=0
    for i in range(len(bone_struct)):
        keypoint = bone_struct[i]
        x= keypoint['x']
        y= keypoint['y']
        if x != 0 and y != 0:
            min_x = x if x<min_x else min_x
            min_y = y if y < min_y else min_y
            max_x = x if x > max_x else max_x
            max_y = y if y > max_y else max_y
    return (min_x-25,min_y-25,max_x-min_x+25,max_y-min_y+25)

def check_bone_struct(bone_struct):
    count=0
    for i in range(len(bone_struct)):
        keypoint = bone_struct[i]
        x= keypoint['x']
        y= keypoint['y']
        if x == 0 and y == 0:
            count=count+1
    if count > 5:
        return  False
    return  True

def map_bone_struct_to_bbox(poseKeypoints_data, boxes  ):
    map=np.zeros(len(poseKeypoints_data))
    print("sfsd")
    for i in range(len(map)):
        map[i]=-1
    for i, newbox in enumerate(boxes):
        min_x=newbox[0]
        min_y=newbox[1]
        max_x=newbox[0] + newbox[2]
        max_y=newbox[1] + newbox[3]

        max_inside=-1
        inside=-1
        for j in range(len(poseKeypoints_data)):
            if map[j]==-1:
                bone_struct=poseKeypoints_data[j]
                count=0
                for k in range(len(bone_struct)):
                    if bone_struct[k]['x'] >= min_x and bone_struct[k]['x'] <= max_x and bone_struct[k]['y'] >= min_y and bone_struct[k]['y']<=max_y:
                        count=count+1
                if count>max_inside:
                    max_inside=count
                    inside = j
        if inside != -1:
            map[inside]=i

    new_boxes=[]
    for i, newbox in enumerate(boxes):
        for j in range(len(poseKeypoints_data)):
            if map[j]==i:
                bone_struct=poseKeypoints_data[j]
                bbox = box_from_bonestruct(bone_struct)
                new_boxes.append(bbox)

    return map,new_boxes

def map_bone_struct_to_bbox(poseKeypoints_data, boxes  ):

    
    number_person=len(poseKeypoints_data)
    number_keypoint=len(poseKeypoints_data[0])
    map=np.zeros(number_person)
    box_in_use=np.zeros(len(boxes))

    print("sfsd")
    for i in range(len(map)):
        map[i]=-1
    for i in range(number_person):
        bone_struct=poseKeypoints_data[i]
        max_inside=2
        inside=-1
        for j, newbox in enumerate(boxes):
            if box_in_use[j]==0:
                count=0
                min_x=newbox[0]
                min_y=newbox[1]
                max_x=newbox[0] + newbox[2]
                max_y=newbox[1] + newbox[3]
                for k in range(number_keypoint):
                    if bone_struct[k]['x'] > min_x and bone_struct[k]['x'] < max_x and bone_struct[k]['y'] > min_y and bone_struct[k]['y']<max_y:
                        count=count+1
                if count>max_inside:
                    if i==number_person-1:
                        max_inside=count
                        inside = j
                    else:
                        wrong_box=False
                        for n in range(i+1,number_person):
                            bone_struct_n=poseKeypoints_data[n]
                            count_n=0
                            for k in range(number_keypoint):
                                if bone_struct_n[k]['x'] >= min_x and bone_struct_n[k]['x'] <= max_x and bone_struct_n[k]['y'] >= min_y and bone_struct_n[k]['y']<=max_y:
                                    count_n=count_n+1
                            if count_n > count :
                                wrong_box = True
                        if wrong_box == False:
                            max_inside=count
                            inside = j
        map[i]=inside


    return map

def convert_float32_to_float(poseKeypoints_data_video):
    for i in range(len(poseKeypoints_data_video)):
        for k in range(len(poseKeypoints_data_video[i])):
            for j in range(len(poseKeypoints_data_video[i][k])):
                poseKeypoints_data_video[i][k][j]['x']= float(poseKeypoints_data_video[i][k][j]['x'])
                poseKeypoints_data_video[i][k][j]['y'] = float(poseKeypoints_data_video[i][k][j]['y'])
    return poseKeypoints_data_video


def convert_float32_to_int(poseKeypoints_data_video):
    for i in range(len(poseKeypoints_data_video)):
        for k in range(len(poseKeypoints_data_video[i])):
            for j in range(len(poseKeypoints_data_video[i][k])):
                poseKeypoints_data_video[i][k][j]['x']= int(poseKeypoints_data_video[i][k][j]['x'])
                poseKeypoints_data_video[i][k][j]['y'] = int(poseKeypoints_data_video[i][k][j]['y'])
    return poseKeypoints_data_video

def save_bone_to_json(poseKeypoints_data_video, output_file):
    print(len(poseKeypoints_data_video))

    print(len(poseKeypoints_data_video[1][0]))

    for i in range(len(poseKeypoints_data_video)):

        for j in range(len(poseKeypoints_data_video[i][0])):
            poseKeypoints_data_video[i][0][j]['x']= float(poseKeypoints_data_video[i][0][j]['x'])
            poseKeypoints_data_video[i][0][j]['y'] = float(poseKeypoints_data_video[i][0][j]['y'])

    print(poseKeypoints_data_video)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(poseKeypoints_data_video, outfile)
