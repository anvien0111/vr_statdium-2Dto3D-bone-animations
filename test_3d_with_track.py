import sys
import cv2
import os
from sys import platform
import argparse
import json
import math
import numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))
from convert_data import *
from openpose_track import *



def main():

    config=Config()
    ops_video_xy=OpenPose_Track_Video(config,'duongxy2')
    ops_video_xy.check_openpose_lib()
    out_keypoint_xy, output_map_xy = ops_video_xy.gen_keypoint()
    poseKeypoints_data_video_xy=convert_data_openpose_to_unity(out_keypoint_xy,output_map_xy)
    print('log xy: ', poseKeypoints_data_video_xy)

    ops_video_yz = OpenPose_Track_Video(config, 'duongyz2')
    ops_video_yz.check_openpose_lib()
    out_keypoint_yz, output_map_yz = ops_video_yz.gen_keypoint()
    poseKeypoints_data_video_yz=convert_data_openpose_to_unity(out_keypoint_yz,output_map_yz)
    print('log yz: ', poseKeypoints_data_video_yz)

    data_3D=convert2Dto3D_PoseKeypoints_data_video(poseKeypoints_data_video_xy,poseKeypoints_data_video_yz)
    with open("output/videos/duong3D2.json", 'w', encoding='utf-8') as outfile:
        json.dump(data_3D, outfile)

main()