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
from random import randint
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



    def check_openpose_lib(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
        return  op

    def set_params(self):
        params = dict()
        # params["logging_level"] = 1
        #params["output_resolution"] = "1280x720"
        params["net_resolution"] = self.config.net_resolution
        params["model_pose"] = self.config.model_pose
        params["number_people_max"]=self.config.number_people_max
        #params['identification'] = True
        # params["alpha_pose"] = 0.6
        # params["scale_gap"] = 0.3
        # params["scale_number"] = 1
        params["render_threshold"] =self.config.render_threshold
        # # If GPU version is built, and multiple GPUs are available, set the ID here
        # params["num_gpu_start"] = 0
        # params["disable_blending"] = False
        # # Ensure you point to the correct path where models are located
        params["model_folder"] = self.config.model_folder

        return params

    def resize_bone(self,bone_struct, fixed_bone_size):

        cur_bone_size = cal_size_bone(bone_struct)
        if cur_bone_size == 0:
            cur_bone_size = self.g_cur_bone_size
        self.g_cur_bone_size = cur_bone_size
        print("cur_bone_size: ", cur_bone_size)
        if cur_bone_size != 0:
            for i in range(len(bone_struct)):
                bone_struct[i]['x'] = np.float32(bone_struct[i]['x'] * fixed_bone_size / cur_bone_size)
                bone_struct[i]['y'] = np.float32(bone_struct[i]['y'] * fixed_bone_size / cur_bone_size)

    def gen_keypoint(self):
        params = self.set_params()

        multiTracker=cv2.MultiTracker_create()
        bboxes = []
        colors = [(255,0,0),(0,255,0)]

        try:
            # Starting OpenPose
            opWrapper = self.op.WrapperPython()
            opWrapper.configure(params)
            opWrapper.start()

            stream = cv2.VideoCapture(self.input_video)
            #stream.set(cv2.CAP_PROP_FPS, 2)
            frame_width = int(stream.get(3))
            frame_height = int(stream.get(4))
            out = cv2.VideoWriter(self.output_vdieo, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                  (frame_width, frame_height))
            # Process Video
            poseKeypoints_data_video = []
            map_bone_video=[]
            while True:
                datum = self.op.Datum()
                ret, img = stream.read()
                if ret == False:
                   break

                datum.cvInputData = img
                opWrapper.emplaceAndPop([datum])


                # Display Image
                #print("Body keypoints: \n" + str(datum.poseKeypoints) + str(type(datum.poseKeypoints)))

                poseKeypoints_data = []
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
                        #self.resize_bone(bone_struct,self.config.fixed_bone_size)
                        poseKeypoints_data.append(bone_struct)
                        print("cal_size: ",cal_size_bone(bone_struct))
                        #draw_bone(bone_struct,cv2,show_result)
                    if len(bboxes) == 0:
                        for bst in poseKeypoints_data:
                            bbox = box_from_bonestruct(bst)
                            bboxes.append(bbox)
                            colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))

                        for bbox in bboxes:
                            multiTracker.add(cv2.TrackerCSRT_create(), img, bbox)





                    poseKeypoints_data_video.append(poseKeypoints_data)
                    print(len(poseKeypoints_data_video))
                    # if len(poseKeypoints_data_video) == 5:
                    #     print("IMPM")
                    #     multiTracker.add(cv2.TrackerCSRT_create(), img, (100, 300, 500, 700))
                    #     colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
                success, boxes = multiTracker.update(img)
                map_bone= map_bone_struct_to_bbox(poseKeypoints_data,boxes)
                map_bone_video.append(map_bone)



                for  i in range(len(poseKeypoints_data)):

                    if map_bone[i]==-1 and check_bone_struct(poseKeypoints_data[i]):
                        bbox = box_from_bonestruct(poseKeypoints_data[i])
                        bboxes.append(bbox)
                        color=(randint(0, 255), randint(0, 255), randint(0, 255))
                        colors.append(color)
                        multiTracker.add(cv2.TrackerCSRT_create(), img, bbox)
                        map_bone[i]=len(colors)-1
                    draw_bone_with_color(poseKeypoints_data[i],cv2,show_result,colors[int(map_bone[i])])



                for i, newbox in enumerate(boxes):
                    p1 = (int(newbox[0]), int(newbox[1]))
                    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                    print(p1, p2)
                    cv2.rectangle(show_result, p1, p2, colors[i], 2, 1)


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


            return [poseKeypoints_data_video, map_bone_video]
        except Exception as e:
            print(e)
        #sys.exit(-1)

def main():
    config=Config()
    ops_video=OpenPose_Video(config,'ducu')

    ops_video.check_openpose_lib()

    out_keypoint,output_map=ops_video.gen_keypoint()
    print(out_keypoint)
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

    with open(ops_video.output_json, 'w', encoding='utf-8') as outfile:
        json.dump(poseKeypoints_data_video, outfile)
    #save_bone_to_json(poseKeypoints_data_video, self.output_json)


main()