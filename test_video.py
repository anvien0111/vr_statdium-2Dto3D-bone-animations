import sys
import cv2
import os
from sys import platform
import argparse
import json
import math
import numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))
from openpose_video import *


def main():

    config = Config()
    ops_video = OpenPose_Video(config, 'fuk')

    ops_video.check_openpose_lib()

    out_keypoint = ops_video.gen_keypoint()

main()