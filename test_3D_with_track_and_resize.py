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

def need_change_curbone_size(curbone_size,cur_bone_struct,prev_bone_struct):

def recalculate_curbone_size(old_cur_bone_size,cur_bone_struct,prev_bone_struct):


def resize_poseKeypoints(poseKeypoints_data_video, fixed_bone_size):
    for i in range(len(poseKeypoints_data_video)):

        cur_bone_size = cal_size_bone(poseKeypoints_data_video[i]['data2d'][0]['data'])
        print("cur_bone_size: ", cur_bone_size)
        for k in range(len(poseKeypoints_data_video[i]['data2d'])):
            for l in range(len(poseKeypoints_data_video[i]['data2d'][k]['data'])):
                poseKeypoints_data_video[i]['data2d'][k]['data'][l]['x'] = int(np.float32(
                    poseKeypoints_data_video[i]['data2d'][k]['data'][l]['x'] * fixed_bone_size / cur_bone_size))
                poseKeypoints_data_video[i]['data2d'][k]['data'][l]['y'] = int(np.float32(
                    poseKeypoints_data_video[i]['data2d'][k]['data'][l]['y'] * fixed_bone_size / cur_bone_size))

    # if self.cur_bone_size == 0:
    #     self.cur_bone_size = self.g_cur_bone_size
    # self.g_cur_bone_size = self.cur_bone_size

    # if self.cur_bone_size != 0:
    #     for i in range(len(bone_struct)):
    #         bone_struct[i]['x'] = np.float32(bone_struct[i]['x'] * fixed_bone_size / self.cur_bone_size)
    #         bone_struct[i]['y'] = np.float32(bone_struct[i]['y'] * fixed_bone_size / self.cur_bone_size)


def main():
    config = Config()
    # ops_video_xy=OpenPose_Track_Video(config,'ducxy_cut')
    # ops_video_xy.check_openpose_lib()
    # out_keypoint_xy, output_map_xy = ops_video_xy.gen_keypoint()
    # poseKeypoints_data_video_xy=convert_data_openpose_to_unity(out_keypoint_xy,output_map_xy)
    # print('log xy: ', poseKeypoints_data_video_xy[0]['data2d'][0]['data'])
    # resize_poseKeypoints(poseKeypoints_data_video_xy,100)
    # print('log xy: ', poseKeypoints_data_video_xy[0]['data2d'][0]['data'])

    config.net_resolution = "1280x720"

    ops_video_yz = OpenPose_Track_Video(config, 'ducyz_cut')
    ops_video_yz.check_openpose_lib()
    out_keypoint_yz, output_map_yz = ops_video_yz.gen_keypoint()
    poseKeypoints_data_video_yz = convert_data_openpose_to_unity(out_keypoint_yz, output_map_yz)
    print('log yz: ', poseKeypoints_data_video_yz[0]['data2d'][0]['data'])
    resize_poseKeypoints(poseKeypoints_data_video_yz, 100)
    print('log yz: ', poseKeypoints_data_video_yz[0]['data2d'][0]['data'])

    # for i in range(len(poseKeypoints_data_video_yz)):
    #     for k in range(len(poseKeypoints_data_video_yz[i]['data2d'])):
    #         if poseKeypoints_data_video_yz[i]['data2d'][k]['data'][1]['x']==0 and k>0:
    #             poseKeypoints_data_video_yz[i]['data2d'][k]['data'][1]['x']=poseKeypoints_data_video_yz[i]['data2d'][k-1]['data'][1]['x']
    #         for l in range(len(poseKeypoints_data_video_yz[i]['data2d'][k]['data'])):
    #             if poseKeypoints_data_video_yz[i]['data2d'][k]['data'][l]['x']==0 :
    #                 poseKeypoints_data_video_yz[i]['data2d'][k]['data'][l]['x']=poseKeypoints_data_video_yz[i]['data2d'][k]['data'][1]['x']

    # data_3D=convert2Dto3D_PoseKeypoints_data_video(poseKeypoints_data_video_xy,poseKeypoints_data_video_yz)
    # with open("output/videos/ducxu_cut_3D2.json", 'w', encoding='utf-8') as outfile:
    #     json.dump(data_3D, outfile)
    # print(data_3D)


main()