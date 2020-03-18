# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import json
import numpy as np
# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append('/usr/local/python')
# from openpose import pyopenpose as op

sys.path.append(dir_path + '/../openpose/build/python/openpose/Release');
os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../openpose/build/x64/Release;' + dir_path + '/../openpose/build/bin;'
import pyopenpose as op


# Flags
parser = argparse.ArgumentParser()
# parser.add_argument("--image_path", default="/input/images/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
args = parser.parse_known_args()

def draw_bone_line(bone_struct,cv2,image,kp1,kp2):
    if bone_struct[kp1]['x']!=0 and bone_struct[kp1]['y']!=0 and bone_struct[kp2]['x']!=0 and bone_struct[kp2]['y']!=0:
        #print("IM HERE")
        cv2.line(image,(np.float32(bone_struct[kp1]['x']),np.float32(bone_struct[kp1]['y'])),(np.float32(bone_struct[kp2]['x']),np.float32(bone_struct[kp2]['y'])),(255,0,0),5)

def draw_bone(bone_struct,cv2,image):
    cv2.circle(image, (100, 100), 5, (0, 0, 0), -1)
    print(bone_struct)
    for i in range(len(bone_struct)):
        # for poseKeypoint in datum.poseKeypoints[0]:
        keypoint = bone_struct[i]
        x_draw =  np.float32(keypoint['x'])
        y_draw = np.float32(keypoint['y'])
        # cv2.putText(show_result, '(' + str(int(x_draw)) + ',' + str(int(y_draw)) + ')', (x_draw, y_draw),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 0), 1, cv2.LINE_AA)
        print(keypoint)
        #try:
        cv2.circle(image, (x_draw, y_draw), 5, (0, 0, 0), -1)
        #except e:
           # print(e)
        print("XXXX")
    #bone_struct_line=[[0,1],[0,15],[0,16],[1,2],[1,5],[1,8],[2,3],[3,4],[5,6],[6,7],[8,9],[8,12],[9,10],[10,11],[12,13],[13,14],[15,17],[16,18]]
    bone_struct_line=[[0,1],[0,14],[0,15],[1,2],[1,5],[1,8], [1,11],[2,3],[3,4],[5,6],[6,7],[8,9],[9,10],[11,12],[12,13],[14,16],[15,17]]
    for line in bone_struct_line:
        #print(line[0],line[1])
        print("I'm Here")
        draw_bone_line(bone_struct,cv2,image,line[0],line[1])
# Custom Params (refer to include/openpose/flags.hpp for more parameters)
def set_params():
    params = dict()
    # params["logging_level"] = 3
    # params["output_resolution"] = "-1x-1"
    params["net_resolution"] = "-1x368"
    params["model_pose"] = "COCO"
    # params["alpha_pose"] = 0.6
    # params["scale_gap"] = 0.3
    # params["scale_number"] = 1
    params["number_people_max"]=1
    # params["render_threshold"] = 0.05
    # # If GPU version is built, and multiple GPUs are available, set the ID here
    # params["num_gpu_start"] = 0
    # params["disable_blending"] = False
    # # Ensure you point to the correct path where models are located
    params["model_folder"] = "../openpose/models/"
    return params
params = set_params()

# Construct it from system arguments
# op.init_argv(args[1])
# oppython = op.OpenposePython()

try:
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    stream = cv2.VideoCapture(0)

    # Process Video
    poseKeypoints_data = []
    while True:

        datum = op.Datum()
        ret, img = stream.read()


        datum.cvInputData = img
        opWrapper.emplaceAndPop([datum])
        poseKeypoints_data = []
        show_result = datum.cvOutputData
        # Display Image
        print("Body keypoints: \n" + str(datum.poseKeypoints))
        if len(datum.poseKeypoints.shape)>1:
            for poseKeypoints in datum.poseKeypoints:
                keypoints = dict()
                for i in range(len(poseKeypoints)):
                    keypoint = dict()
                    keypoint['x'] = str(poseKeypoints[i][0])
                    keypoint['y'] = str(poseKeypoints[i][1])
                    keypoint['confident'] = str(poseKeypoints[i][2])
                    keypoints[i] = keypoint
                poseKeypoints_data.append(keypoints)
                draw_bone(keypoints,cv2,show_result)
                cv2.circle(show_result, (100, 100), 5, (0, 0, 0), -1)
        cv2.imshow("OpenPose DEMO CAMERA", show_result)

        key = cv2.waitKey(1)
        if key ==ord('q'):
            break
    with open('output/data_came2.json', 'w', encoding='utf-8') as outfile:
        json.dump(poseKeypoints_data, outfile)
    stream.release()
    cv2.destroyAllWindows()
except Exception as e:
    # print(e)
    sys.exit(-1)
