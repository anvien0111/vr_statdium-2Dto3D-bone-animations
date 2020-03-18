
#from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def convert_float32_to_int(poseKeypoints_data_video):
    for i in range(len(poseKeypoints_data_video)):
        for k in range(len(poseKeypoints_data_video[i])):
            for j in range(len(poseKeypoints_data_video[i][k])):
                poseKeypoints_data_video[i][k][j]['x']= int(poseKeypoints_data_video[i][k][j]['x'])
                poseKeypoints_data_video[i][k][j]['y'] = int(poseKeypoints_data_video[i][k][j]['y'])
    return poseKeypoints_data_video

def convert_data_openpose_to_unity(out_keypoint,output_map):
    out_keypoint = convert_float32_to_int(out_keypoint)
    number_object = len(out_keypoint[0])
    poseKeypoints_data_video = []
    for i in range(0, number_object):
        object_data = []
        for j in range(0, len(out_keypoint)):
            for k in range(0, len(out_keypoint[j])):
                if (int(output_map[j][k]) == i):
                    object_data.append({'id': i, 'frame': j, 'data': out_keypoint[j][k]})
        poseKeypoints_data_video.append({"id": i, 'data2d': object_data})

def convert2Dto3D_bone_struct(bone_struct_xy, bone_struct_zy):
    # bone_struct={0: {'x': 354.42596435546875, 'y': 58.433895111083984}, 1: {'x': 354.4095153808594, 'y': 101.20509338378906}..}
    bone_struct = dict()
    for i in range(len(bone_struct_xy)):
        keypoint = dict()
        keypoint['x'] = bone_struct_xy[i]['x']
        keypoint['y'] = bone_struct_xy[i]['y']
        keypoint['z'] = bone_struct_zy[i]['x']
        bone_struct[i]=keypoint
    return  bone_struct

def convert2Dto3D_bone_data(bone_data_xy,bone_data_zy):
    size_data = len(bone_data_xy)
    if size_data>len(bone_data_zy):
        size_data=len(bone_data_zy)
    output_3D=[]
    for i in range(size_data):
        bone_struct_xy=bone_data_xy[i][0]
        bone_struct_zy=bone_data_zy[i][0]
        bone_struct_3D=convert2Dto3D_bone_struct(bone_struct_xy,bone_struct_zy)
        output_3D.append(bone_struct_3D)
    return  output_3D

def convert2Dto3D_bone_data_with_track(bone_data_xy,bone_data_zy,id):
    size_data = len(bone_data_xy)
    if size_data>len(bone_data_zy):
        size_data=len(bone_data_zy)
    object_data_3D=[]
    for j in range(size_data):
        bone_struct_xy=bone_data_xy[j]['data']
        bone_struct_zy=bone_data_zy[j]['data']
        bone_struct_3D=convert2Dto3D_bone_struct(bone_struct_xy,bone_struct_zy)
        object_data_3D.append({'id': id, 'frame': j, 'data': bone_struct_3D})

    return  object_data_3D

def convert2Dto3D_PoseKeypoints_data_video(poseKeypoints_data_video_xy,poseKeypoints_data_video_yz):
    pairs=[[0,0],[1,1]]
    poseKeypoints_data_video_3D=[]
    for i in range(len(pairs)):
        person_xy=pairs[i][0]
        person_yz=pairs[i][1]
        bone_data_xy=poseKeypoints_data_video_xy[person_xy]['data2d']
        bone_data_yz=poseKeypoints_data_video_yz[person_yz]['data2d']
        bone_data_xyz=convert2Dto3D_bone_data(bone_data_xy,bone_data_yz,i)
        poseKeypoints_data_video_3D.append({"id": i, 'data3d': bone_data_xyz})
    return  poseKeypoints_data_video_3D






#def cut_video(name, start_time, end_time,output):
#    ffmpeg_extract_subclip(name, start_time, end_time, tarrgetname="/input/videos/"+output)