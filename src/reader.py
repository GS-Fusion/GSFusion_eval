#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import json
from typing import NamedTuple

import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from src.utils import focal2fov
from src.colmap import qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary


class CameraInfo(NamedTuple):
    R: np.array
    T: np.array
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    FovX: float
    FovY: float
    image_path: str


class SceneInfo(NamedTuple):
    train_cameras: list
    test_cameras: list


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        R = qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec)

        if intr.model=="PINHOLE":
            fx = intr.params[0]
            fy = intr.params[1]
            cx = intr.params[2]
            cy = intr.params[3]
            FovX = focal2fov(fx, width)
            FovY = focal2fov(fy, height)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))

        cam_info = CameraInfo(R=R, T=T, fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height, FovX=FovX, FovY=FovY, image_path=image_path)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapSceneInfo(path, images, eval):
    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_path)
    train_cam_infos = cam_infos
    test_cam_infos = []

    scene_info = SceneInfo(train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos)
    return scene_info


def readScanNetppSceneInfo(path, images, eval):
    with open(os.path.join(path, "train_test_lists.json"), 'r') as split_f:
        splits = json.load(split_f)

    with open(os.path.join(path, "nerfstudio/transforms_undistorted_2.json"), 'r') as gt_f:
        gt = json.load(gt_f)
    
    fx = gt["fl_x"]
    fy = gt["fl_y"]
    cx = gt["cx"]
    cy = gt["cy"]
    width = gt['w']
    height = gt['h']
    FovX = focal2fov(fx, width)
    FovY = focal2fov(fy, height)

    gt["frames"] = sorted(gt["frames"], key = lambda x : x["file_path"])
    if eval:
        gt["test_frames"] = sorted(gt["test_frames"], key = lambda x : x["file_path"])

    if eval:
        split_types = ["train", "test"]
    else:
        split_types = ["train"]

    train_cam_infos = []
    test_cam_infos = []
    for split_type in split_types:
        print(f"Reading {split_type} cameras")

        cam_infos = []
        if split_type == "train":
            frames = gt["frames"]
        else:
            frames = gt["test_frames"]

        for i, image_name in tqdm(enumerate(splits[split_type])):
            assert image_name == frames[i]["file_path"], "Unmatched images and poses!"

            image_path = os.path.join(path, images, image_name)

            # Coordinate convensions
            # ScanNet++ uses the OpenGL/Blender (and original NeRF) coordinate convention for cameras. 
            # +X is right, +Y is up, and +Z is pointing back and away from the camera. -Z is the look-at direction. 
            # Other codebases may use the COLMAP/OpenCV convention, where the Y and Z axes are flipped from ours but the +X axis remains the same.
            T_C2W = np.array(frames[i]["transform_matrix"])
            P = np.eye(4)
            P[1, 1] = -1
            P[2, 2] = -1
            T_C2W = np.dot(T_C2W, P)
            T_W2C = np.linalg.inv(T_C2W)
            R = T_W2C[:3, :3]
            T = T_W2C[:3, 3]

            cam_info = CameraInfo(R=R, T=T, fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height, FovX=FovX, FovY=FovY, image_path=image_path)
            cam_infos.append(cam_info)

        if split_type == "train":
            train_cam_infos = cam_infos
        elif split_type == "test":
            test_cam_infos = cam_infos

    scene_info = SceneInfo(train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos)
    return scene_info


def readARKitScenesInfo(path, images, eval):
    if (not os.path.exists("/tmp/arkitscenes_gt.txt")):
        raise FileExistsError("There is no such file: /tmp/arkitscenes_gt.txt")
    with open("/tmp/arkitscenes_gt.txt", 'r') as file: 
        gt_infos = file.readlines()
    
    intrinsics_fname = gt_infos[2].strip().split()[2]
    intrinsics_fname = intrinsics_fname.replace("lowres_wide", "lowres_wide_intrinsics")
    intrinsics_fname = intrinsics_fname.replace("png", "pincam")
    with open(intrinsics_fname, 'r') as file:
        intrinsics = file.readline()
        intrinsics = intrinsics.strip().split()

    width = int(intrinsics[0])
    height = int(intrinsics[1])
    fx = float(intrinsics[2])
    fy = float(intrinsics[3])
    cx = float(intrinsics[4])
    cy = float(intrinsics[5])
    FovX = focal2fov(fx, width)
    FovY = focal2fov(fy, height)

    train_cam_infos = []
    for gt_info in gt_infos:
        if gt_info.startswith('#'):
            continue
        gt_info = gt_info.strip().split()
        image_path = gt_info[2]

        tx = float(gt_info[4])
        ty = float(gt_info[5])
        tz = float(gt_info[6])
        qx = float(gt_info[7])
        qy = float(gt_info[8])
        qz = float(gt_info[9])
        qw = float(gt_info[10])

        t_wc = np.array([tx, ty, tz])
        q_wc = np.array([qx, qy, qz, qw])
        
        R_wc = Rotation.from_quat(q_wc).as_matrix()
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc

        cam_info = CameraInfo(R=R_cw, T=t_cw, fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height, FovX=FovX, FovY=FovY, image_path=image_path)
        train_cam_infos.append(cam_info)
    
    scene_info = SceneInfo(train_cameras=train_cam_infos,
                           test_cameras=[])
    return scene_info


def readReplicaSceneInfo(path, images, eval):
    parent_path = os.path.dirname(path)
    intrinsics_path = os.path.join(parent_path, "cam_params.json")
    with open(intrinsics_path, 'r') as file:
        intrinsics = json.load(file)
    width = intrinsics["camera"]["w"]
    height = intrinsics["camera"]["h"]
    fx = intrinsics["camera"]["fx"]
    fy = intrinsics["camera"]["fy"]
    cx = intrinsics["camera"]["cx"]
    cy = intrinsics["camera"]["cy"]
    FovX = focal2fov(fx, width)
    FovY = focal2fov(fy, height)

    data_path = os.path.join(path, images)
    all_filenames = os.listdir(data_path)
    frame_filenames = [f for f in all_filenames if f.startswith("frame")]
    frame_filenames.sort()

    gt_path = os.path.join(path, "traj.txt")
    if (not os.path.exists(gt_path)):
        raise FileExistsError(f"There is no such file: {gt_path}")   
    with open(gt_path, 'r') as file: 
        gt_infos = file.readlines()
    
    assert len(gt_infos) == len(frame_filenames), "Unmatched images and poses!"
    
    train_cam_infos = []
    for i, gt_info in enumerate(gt_infos):
        image_path = os.path.join(data_path, frame_filenames[i])
        gt_info = gt_info.strip().split()

        R_wc = np.array([[float(gt_info[0]), float(gt_info[1]), float(gt_info[2])],
                         [float(gt_info[4]), float(gt_info[5]), float(gt_info[6])],
                         [float(gt_info[8]), float(gt_info[9]), float(gt_info[10])]])
        R_wc = Rotation.from_matrix(R_wc).as_matrix()
        t_wc = np.array([float(gt_info[3]), float(gt_info[7]), float(gt_info[11])])

        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc

        cam_info = CameraInfo(R=R_cw, T=t_cw, fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height, FovX=FovX, FovY=FovY, image_path=image_path)
        train_cam_infos.append(cam_info)
    
    scene_info = SceneInfo(train_cameras=train_cam_infos,
                           test_cameras=[])
    return scene_info


sceneLoadTypeCallbacks = {
    "colmap": readColmapSceneInfo,
    "scannetpp": readScanNetppSceneInfo,
    "replica": readReplicaSceneInfo,
    "arkitscenes": readARKitScenesInfo
}
