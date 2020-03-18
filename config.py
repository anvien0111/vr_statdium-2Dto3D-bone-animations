class Config(object):
    logging_level = 1
    output_resolution = "-1x-1"
    net_resolution = "-1x64"
    model_pose = "BODY_25"
    alpha_pose = 1
    scale_gap = 0.3
    scale_number = 1
    model_folder = "../openpose/models/"
    fixed_bone_size = 100
    number_people_max=-1
    render_threshold=0.5
    disable_blending = False
    colors = [(0,153,51),(255,0,0),(0,204,153),(0,102,255),(255,0,255),(204,51,0),(153,102,51),(0,102,0),(51,51,153),(153,204,255),(102,0,51),(153,153,102)]

