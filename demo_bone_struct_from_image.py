# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import json

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/../openpose/build/python/openpose/Release');
os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../openpose/build/x64/Release;' + dir_path + '/../openpose/build/bin;'
import pyopenpose as op

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default="input/images/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
def set_params():
    params = dict()
    # params["logging_level"] = 3
    # params["output_resolution"] = "-1x-1"
    params["net_resolution"] = "-1x368"
    params["model_pose"] = "MPI"
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

# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1])-1: next_item = args[1][i+1]
    else: next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-','')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-','')
        if key not in params: params[key] = next_item

# Construct it from system arguments
# op.init_argv(args[1])
# oppython = op.OpenposePython()

try:
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)

    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])

    # Display Image
    print("Body keypoints: \n" + str(datum.poseKeypoints))

    poseKeypoints_data = []
    for poseKeypoints in datum.poseKeypoints:
        keypoints = dict()
        for i in range(len(poseKeypoints)):
            keypoint = dict()
            keypoint['x'] = str(poseKeypoints[i][0])
            keypoint['y'] = str(poseKeypoints[i][1])
            keypoint['confident'] = str(poseKeypoints[i][2])
            keypoints[i] = keypoint
        poseKeypoints_data.append(keypoints)

    with open('output/data_image.json', 'w', encoding='utf-8') as outfile:
        json.dump(poseKeypoints_data, outfile)


    show_result = datum.cvOutputData
    x_draw = datum.poseKeypoints[0][0][0]
    y_draw = datum.poseKeypoints[0][0][1]
    cv2.putText(show_result, str(int(x_draw)) + ', ' + str(int(y_draw)), (x_draw, y_draw), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                (255, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(show_result, (x_draw, y_draw), 5, (0, 0, 255), -1)
    cv2.imshow("OpenPose DEMO IMAGE", show_result)

    cv2.waitKey(0)
except Exception as e:
    print(e)
    sys.exit(-1)
