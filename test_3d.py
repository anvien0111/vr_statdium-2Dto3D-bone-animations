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
from openpose_video import *

def main():

    config=Config()
    ops_video_xy=OpenPose_Video(config,'duongxy2')
    ops_video_xy.check_openpose_lib()
    out_keypoints_xy=ops_video_xy.gen_keypoint()
    print('log xy: ', out_keypoints_xy)

    ops_video_yz = OpenPose_Video(config, 'duongyz2')
    ops_video_yz.check_openpose_lib()
    out_keypoints_yz = ops_video_yz.gen_keypoint()
    print('log xy: ', out_keypoints_yz)

    data_3D=convert2Dto3D_bone_data(out_keypoints_xy,out_keypoints_yz)
    with open("output/videos/duong3D2.json", 'w', encoding='utf-8') as outfile:
        json.dump(data_3D, outfile)

main()