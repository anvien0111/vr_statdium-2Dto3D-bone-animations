# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import json
import math
import numpy as np


from config  import *
from bone_struct_data import *

class OpenPose_Video(object):
    def __init__(self,config,input_video_name):
        self.config = config
        self.input_video = "input/videos/" + input_video_name + ".mp4"
        self.output_vdieo = "output/videos/" + input_video_name + "_out.avi"
        self.output_json= "output/videos/" + input_video_name + "_out.json"
        self.op = self.check_openpose_lib()
        self.parser = argparse.ArgumentParser()
        self.args= self.parser.parse_known_args()
        self.g_cur_bone_size = self.config.fixed_bone_size
        self.cur_bone_size=0


    def check_openpose_lib(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
        return  op

    def set_params(self):
        params = dict()
        # params["logging_level"] = 1
        # params["output_resolution"] = "-1x-1"
        params["net_resolution"] = self.config.net_resolution
        params["model_pose"] = self.config.model_pose
        params["number_people_max"]=1

        # params["alpha_pose"] = 0.6
        # params["scale_gap"] = 0.3
        # params["scale_number"] = 1
        # params["render_threshold"] = 0.05
        # # If GPU version is built, and multiple GPUs are available, set the ID here
        # params["num_gpu_start"] = 0
        params['tracking']=0

        #params['identification'] = True
        # params["disable_blending"] = False
        # # Ensure you point to the correct path where models are located
        params["model_folder"] = self.config.model_folder

        return params

    def resize_bone(self,bone_struct, fixed_bone_size):
        if self.cur_bone_size==0:
            self.cur_bone_size = cal_size_bone(bone_struct)
        if self.cur_bone_size == 0:
            self.cur_bone_size = self.g_cur_bone_size
        self.g_cur_bone_size = self.cur_bone_size
        print("cur_bone_size: ", self.cur_bone_size)
        if self.cur_bone_size != 0:
            for i in range(len(bone_struct)):
                bone_struct[i]['x'] = np.float32(bone_struct[i]['x'] * fixed_bone_size / self.cur_bone_size)
                bone_struct[i]['y'] = np.float32(bone_struct[i]['y'] * fixed_bone_size / self.cur_bone_size)

    def gen_keypoint(self):
        params = self.set_params()

        try:
            # Starting OpenPose
            opWrapper = self.op.WrapperPython()
            opWrapper.configure(params)
            opWrapper.start()

            stream = cv2.VideoCapture(self.input_video)

            frame_width = int(stream.get(3))
            frame_height = int(stream.get(4))
            out = cv2.VideoWriter(self.output_vdieo, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                  (frame_width, frame_height))
            # Process Video
            poseKeypoints_data_video = []
            while True:
                datum = self.op.Datum()
                ret, img = stream.read()
                if ret == False:
                   break

                datum.cvInputData = img
                opWrapper.emplaceAndPop([datum])

                # Display Image
                print("Body keypoints: \n" + str(datum.poseKeypoints) + str(type(datum.poseKeypoints)))

                poseKeypoints_data_frame = []
                show_result = datum.cvOutputData

                if len(datum.poseKeypoints.shape)>1:
                    for poseKeypoints in datum.poseKeypoints:
                        bone_struct = dict()
                        for i in range(len(poseKeypoints)):
                            keypoint = dict()
                            keypoint['x'] = poseKeypoints[i][0]
                            keypoint['y'] = poseKeypoints[i][1]
                            #keypoint['c'] = str(poseKeypoints[i][2])
                            bone_struct[i] = keypoint
                        self.resize_bone(bone_struct,self.config.fixed_bone_size)
                        poseKeypoints_data_frame.append(bone_struct)
                        print("cal_size: ",cal_size_bone(bone_struct))
                        draw_bone(bone_struct,cv2,show_result)

                    poseKeypoints_data_video.append(poseKeypoints_data_frame)
                    print(len(poseKeypoints_data_video))
                cv2.imshow("OpenPose OUPUT VIDEO", show_result)
                out.write(show_result)

                print("YYYYYYYYYYYYYYYYYYYYYYYYYYEEEEE")
                key = cv2.waitKey(1)
                if key == ord('q') :
                    break
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

            stream.release()
            out.release()
            cv2.destroyAllWindows()

            print(poseKeypoints_data_video)
            save_bone_to_json(poseKeypoints_data_video,self.output_json)
            return poseKeypoints_data_video
        except Exception as e:
            print(e)
        #sys.exit(-1)

# def main():
#     config=Config()
#     ops_video=OpenPose_Video(config,'fuk')
#
#     ops_video.check_openpose_lib()
#
#     out_keypoint=ops_video.gen_keypoint()


#main()