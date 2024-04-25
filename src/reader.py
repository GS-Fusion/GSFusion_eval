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
from PIL import Image

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
    image: np.array
    image_path: str
    image_name: str


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
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(R=R, T=T, fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height, FovX=FovX, FovY=FovY,
                              image=image, image_path=image_path, image_name=image_name)
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
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    train_cam_infos = cam_infos
    test_cam_infos = []

    scene_info = SceneInfo(train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos)
    return scene_info


def readScanNetppSceneInfo(path, images, eval):
    with open(os.path.join(path, "train_test_lists.json"), 'r') as split_f:
        splits = json.load(split_f)

    with open(os.path.join(path, "nerfstudio/transforms_undistorted.json"), 'r') as gt_f:
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
        split_types = ["train", "test"]
    else:
        split_types = ["train"]

    cnt = 0
    train_cam_infos = []
    test_cam_infos = []
    for split_type in split_types:
        cam_infos = []
        print(f"Reading {split_type} cameras")
        for image_name in tqdm(splits[split_type]):
            assert image_name == gt["frames"][cnt]["file_path"], "Unmatched images and poses!"

            image_path = os.path.join(path, images, image_name)
            image_name = image_name.split('.')[0]
            image = Image.open(image_path)

            # Coordinate convensions
            # ScanNet++ uses the OpenGL/Blender (and original NeRF) coordinate convention for cameras. 
            # +X is right, +Y is up, and +Z is pointing back and away from the camera. -Z is the look-at direction. 
            # Other codebases may use the COLMAP/OpenCV convention, where the Y and Z axes are flipped from ours but the +X axis remains the same.
            T_C2W = np.array(gt["frames"][cnt]["transform_matrix"])
            P = np.eye(4)
            P[1, 1] = -1
            P[2, 2] = -1
            T_C2W = np.dot(P, np.dot(T_C2W, P.transpose()))
            T_W2C = np.linalg.inv(T_C2W)
            R = T_W2C[:3, :3]
            T = T_W2C[:3, 3]

            cam_info = CameraInfo(R=R, T=T, fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height, FovX=FovX, FovY=FovY,
                                  image=image, image_path=image_path, image_name=image_name)
            cam_infos.append(cam_info)
            cnt += 1

        if split_type == "train":
            train_cam_infos = cam_infos
        elif split_type == "test":
            test_cam_infos = cam_infos

    scene_info = SceneInfo(train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos)
    return scene_info


sceneLoadTypeCallbacks = {
    "colmap": readColmapSceneInfo,
    "scannetpp": readScanNetppSceneInfo
}
