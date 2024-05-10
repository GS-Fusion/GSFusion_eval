import os
from argparse import ArgumentParser

import trimesh
import numpy as np

from src.gs import GaussianModel
from src.utils import chamfer_distance, searchForMaxIteration
from src.arguments import ModelParams, get_combined_args


def main(opt):
    # Load trained Gaussian Splatting model and extract point cloud from GS model
    max_iteration = searchForMaxIteration(os.path.join(opt.model_path, "point_cloud"))
    if args.iteration == -1:
        load_iteration = max_iteration
    else:
        load_iteration = args.iteration
    print("Loading trained model at iteration {}".format(load_iteration))
    gaussians = GaussianModel(opt.sh_degree)
    gaussians.load_ply(os.path.join(opt.model_path,
                                    "point_cloud",
                                    "iteration_" + str(load_iteration),
                                    "point_cloud.ply"))

    pred_pcd = gaussians.get_xyz.data.cpu().numpy()
    pred_pcd = pred_pcd.astype(np.float32)
    print(f"#pts of pred_pcd: {pred_pcd.shape}")

    # Load gt mesh and extract point cloud from gt mesh (1 point per cm2)
    gt_mesh = trimesh.load_mesh(os.path.join(opt.model_path, "mesh", f"mesh_{str(max_iteration)}.ply"))
    gt_area = int(gt_mesh.area * 1e4)

    gt_pcd = gt_mesh.sample(gt_area)
    gt_pcd = gt_pcd.astype(np.float32)
    print(f"#pts of gt_pcd: {gt_pcd.shape}")

    # Compute accuracy: chamfer distance from pred_pcd to gt_pcd
    accuracy, _ = chamfer_distance(pred_pcd, None, gt_pcd, None)
    accuracy2 = accuracy ** 2
    print(f"L1 SCD: {accuracy.mean()}")
    print(f"L2 SCD: {accuracy2.mean()}")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)

    main(model.extract(args))
