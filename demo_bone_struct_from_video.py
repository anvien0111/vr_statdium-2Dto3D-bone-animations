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

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/usr/local/python')
from openpose import pyopenpose as op

# Flags
parser = argparse.ArgumentParser()
# parser.add_argument("--image_path", default="/input/images/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
args = parser.parse_known_args()


g_cur_bone_size = 100

def cal_distance(x1,y1,x2,y2):
    return math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))

def cal_size_bone(bone_struct):
    if bone_struct[1]['x']==0 or bone_struct[1]['y']==0 or bone_struct[8]['x']==0 or bone_struct[8]['y']==0:
        return 0
    return cal_distance(bone_struct[1]['x'],bone_struct[1]['y'],bone_struct[8]['x'],bone_struct[8]['y'])

def resize_bone(bone_struct,fixed_bone_size):
    global g_cur_bone_size
    cur_bone_size=cal_size_bone(bone_struct)
    if cur_bone_size ==0:
        cur_bone_size=g_cur_bone_size
    g_cur_bone_size=cur_bone_size
    print("cur_bone_size: ",cur_bone_size)
    if cur_bone_size!=0:
        for i in range(len(bone_struct)):
            bone_struct[i]['x']=np.float32(bone_struct[i]['x']*fixed_bone_size/cur_bone_size)
            bone_struct[i]['y']=np.float32(bone_struct[i]['y']*fixed_bone_size/cur_bone_size)

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
    bone_struct_line=[[0,1],[0,14],[0,15],[1,2],[1,5],[1,8], [1,11],[2,3],[3,4],[5,6],[6,7],[8,9],[9,10],[11,12],[12,13],[14,16],[15,17]]
    for line in bone_struct_line:
        #print(line[0],line[1])
        draw_bone_line(bone_struct,cv2,image,line[0],line[1])


# Custom Params (refer to include/openpose/flags.hpp for more parameters)
def set_params():
    params = dict()
    #params["logging_level"] = 1
    # params["output_resolution"] = "-1x-1"
    params["net_resolution"] = "-1x368"
    params["model_pose"] = "BODY_25"
    # params["alpha_pose"] = 0.6
    # params["scale_gap"] = 0.3
    # params["scale_number"] = 1
    # params["render_threshold"] = 0.05
    # # If GPU version is built, and multiple GPUs are available, set the ID here
    # params["num_gpu_start"] = 0
    # params["disable_blending"] = False
    # # Ensure you point to the correct path where models are located
    params["model_folder"] = "../openpose/models/"

    return params
params = set_params()
fixed_bone_size = 100
# Construct it from system arguments
# op.init_argv(args[1])
# oppython = op.OpenposePython()

try:
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    stream = cv2.VideoCapture("input/videos/input_model.mp4")

    frame_width = int(stream.get(3))
    frame_height = int(stream.get(4))
    out = cv2.VideoWriter('output/videos/inputSSS.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                          (frame_width, frame_height))
    # Process Video
    poseKeypoints_data_video = []

    while True:

        datum = op.Datum()

        ret, img = stream.read()
        if ret == False:
           break

        datum.cvInputData = img
        opWrapper.emplaceAndPop([datum])

        # Display Image
        print("Body keypoints: \n" + str(datum.poseKeypoints) + str(type(datum.poseKeypoints)))

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
                resize_bone(bone_struct,fixed_bone_size)
                poseKeypoints_data.append(bone_struct)
                print("cal_size: ",cal_size_bone(bone_struct))

                # for i in range(len(bone_struct)):
                #     # for poseKeypoint in datum.poseKeypoints[0]:
                #     keypoint = bone_struct[i]
                #     x_draw = keypoint['x']
                #     y_draw = keypoint['y']
                #     # cv2.putText(show_result, '(' + str(int(x_draw)) + ',' + str(int(y_draw)) + ')', (x_draw, y_draw),
                #     #            cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 0), 1, cv2.LINE_AA)
                #     #cv2.circle(show_result, (x_draw, y_draw), 5, (0, 0, 0), -1)
                draw_bone(bone_struct,cv2,show_result)

            poseKeypoints_data_video.append(poseKeypoints_data)
            print(len(poseKeypoints_data_video))

            # for i in range(len(datum.poseKeypoints[0])):
            # #for poseKeypoint in datum.poseKeypoints[0]:
            #     poseKeypoint=datum.poseKeypoints[0][i]
            #     x_draw = poseKeypoint[0]
            #     y_draw = poseKeypoint[1]
            #     #cv2.putText(show_result, '(' + str(int(x_draw)) + ',' + str(int(y_draw)) + ')', (x_draw, y_draw),
            #     #            cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 0), 1, cv2.LINE_AA)
            #     if i==0:
            #         print('x_draw: ',type(x_draw),' curbone0x: ',type(cur_bone[0]['x']))
            #         cv2.circle(show_result, (cur_bone[0]['x'], y_draw), 5, (0, 0, 0), -1)

        cv2.imshow("OpenPose DEMO VIDEO", show_result)
        out.write(show_result)

        print("YYYYYYYYYYYYYYYYYYYYYYYYYYEEEEE")
        key = cv2.waitKey(1)
        if key == ord('q') :
            print("YYYYYYYYYYYYYYYYYYYYYYYYYY")
            break





    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

    stream.release()
    out.release()
    cv2.destroyAllWindows()

    print(poseKeypoints_data_video)
    with open('output/data_lamtv_example1.json', 'w', encoding='utf-8') as outfile:
        json.dump(poseKeypoints_data_video, outfile)

except Exception as e:
    # print(e)
    sys.exit(-1)
