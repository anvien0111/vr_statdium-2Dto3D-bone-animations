
import sys
import cv2
import os
from sys import platform
import argparse
import json
import math
import numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))
from openpose_track import *

def main():
    config=Config()
    ops_video=OpenPose_Track_Video(config,'ducu')

    ops_video.check_openpose_lib()

    out_keypoint,output_map=ops_video.gen_keypoint()
    #print(out_keypoint)

    out_keypoint=convert_float32_to_int(out_keypoint)
    number_object= len(out_keypoint[0])
    poseKeypoints_data_video=[]
    for i in range(0, number_object):
        object_data=[]
        for j in range(0, len(out_keypoint)):
            for k in range(0, len(out_keypoint[j])):
                if(int(output_map[j][k])==i):
                    object_data.append({'id':i,'frame':j,'data':out_keypoint[j][k]})
        poseKeypoints_data_video.append({"id":i, 'data2d':object_data})

    print(poseKeypoints_data_video)


    outfile=ops_video.output_json
    with open(ops_video.output_json, 'w', encoding='utf-8') as outfile:
        json.dump(poseKeypoints_data_video, outfile)
    #save_bone_to_json(poseKeypoints_data_video, self.output_json)


main()