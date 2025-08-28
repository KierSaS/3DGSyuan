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
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

# =========================
# Data containers
# =========================

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool
    # ↓↓↓ NEW: explicit intrinsics for projection (from COLMAP)
    fx: float
    fy: float
    cx: float
    cy: float

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

# =========================
# Helpers
# =========================

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center
    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, 'vertex')
    PlyData([vertex_element]).write(path)

# =========================
# COLMAP readers
# =========================

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id

        # NOTE: In original repo they store R transposed for CUDA code convenience
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # ---------- intrinsics ----------
        # COLMAP models:
        #   SIMPLE_PINHOLE: [f, cx, cy]
        #   PINHOLE       : [fx, fy, cx, cy]
        if intr.model == "SIMPLE_PINHOLE":
            f  = intr.params[0]
            cx = intr.params[1]
            cy = intr.params[2]
            fx, fy = f, f
        elif intr.model == "PINHOLE":
            fx = intr.params[0]
            fy = intr.params[1]
            cx = intr.params[2]
            cy = intr.params[3]
        else:
            assert False, "Unsupported camera model (need undistorted SIMPLE_PINHOLE or PINHOLE)."

        FovY = focal2fov(fy, height)
        FovX = focal2fov(fx, width)

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list,
                              fx=fx, fy=fy, cx=cx, cy=cy)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

# =========================
# Point cloud mask-filtering
# =========================

def _load_masks_as_dict(masks_dir):
    """Load binary masks (0/255) into a dict: {basename_with_ext: np.uint8[H,W]}"""
    masks = {}
    if not os.path.isdir(masks_dir):
        return masks
    for name in os.listdir(masks_dir):
        p = os.path.join(masks_dir, name)
        if not os.path.isfile(p):
            continue
        try:
            m = Image.open(p).convert("L")
            masks[name] = (np.array(m) > 127).astype(np.uint8)  # 0/1
        except Exception:
            pass
    return masks

def _project_points_to_cam(pts, cam: CameraInfo):
    """
    Project world points (N,3) into pixel coords using COLMAP convention.
    We stored R as transpose of colmap R_wc; original R_wc = R.T; T= tvec.
    X_cam = R_wc * X_world + t
    u = fx * Xc.x / Xc.z + cx ; v = fy * Xc.y / Xc.z + cy
    """
    # Use original world->cam rotation:
    R_wc = cam.R.T  # (3,3)
    t    = cam.T.reshape(3,)

    Xc = (pts @ R_wc.T) + t  # (N,3)
    z  = Xc[:, 2]
    valid = z > 1e-6
    u = cam.fx * (Xc[:, 0] / np.maximum(z, 1e-6)) + cam.cx
    v = cam.fy * (Xc[:, 1] / np.maximum(z, 1e-6)) + cam.cy
    return u, v, valid

def filter_point_cloud_with_masks(pcd: BasicPointCloud,
                                  cam_infos: list,
                                  masks_dir: str,
                                  sample_cams: int = 20,
                                  vis_thresh: float = 0.3):
    """
    Keep points whose multi-view mask-foreground ratio >= vis_thresh.
    - sample_cams: randomly sample up to this many cameras per point (speed/quality tradeoff)
    """
    if not os.path.isdir(masks_dir) or len(pcd.points) == 0:
        return pcd  # nothing to do

    # Build quick lookup for masks by image file name
    # (undistorted images keep same names; ensure your masks dir uses identical names)
    masks = _load_masks_as_dict(masks_dir)
    if not masks:
        return pcd

    N = pcd.points.shape[0]
    pts = pcd.points.astype(np.float64)

    # Choose a subset of cameras to check per point for speed
    rng = np.random.default_rng(0)
    cams = np.array(cam_infos)
    if len(cams) > sample_cams:
        # uniformly sample a fixed subset for all points (fast and simple)
        idx = rng.choice(len(cams), size=sample_cams, replace=False)
        cams = cams[idx]

    keep = np.zeros(N, dtype=bool)
    H_cache = {}
    W_cache = {}

    # Pre-cache sizes for speed
    for c in cams:
        if c.image_name in masks:
            H_cache[c.image_name] = masks[c.image_name].shape[0]
            W_cache[c.image_name] = masks[c.image_name].shape[1]

    # Iterate points (vectorization across all points is possible, but per-point keeps it simpler)
    for i in range(N):
        X = pts[i:i+1, :]  # (1,3)
        vis = 0
        fg  = 0
        for c in cams:
            if c.image_name not in masks:
                continue
            u, v, valid = _project_points_to_cam(X, c)
            if not valid[0]:
                continue
            H = H_cache[c.image_name]; W = W_cache[c.image_name]
            uu = int(round(u[0])); vv = int(round(v[0]))
            if uu < 0 or uu >= W or vv < 0 or vv >= H:
                continue
            vis += 1
            if masks[c.image_name][vv, uu] > 0:
                fg += 1
        if vis > 0 and (fg / vis) >= vis_thresh:
            keep[i] = True

    # If too few kept (e.g., masks misaligned), fall back to original to avoid empty init
    kept = keep.sum()
    if kept < max(1000, int(0.02 * N)):  # safety fallback threshold
        print(f"[MaskFilter] Too few points kept ({kept}/{N}), skip filtering to be safe.")
        return pcd

    print(f"[MaskFilter] Kept {kept}/{N} points ({100.0*kept/N:.2f}%) as foreground.")
    new_points = pcd.points[keep]
    new_colors = pcd.colors[keep]
    new_normals = pcd.normals[keep]
    return BasicPointCloud(points=new_points, colors=new_colors, normals=new_normals)

# =========================
# Scene readers
# =========================

def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    # ---- Read COLMAP DB ----
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    # ---- Optional depth scaling params ----
    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            med_scale = np.median(all_scales[all_scales > 0]) if (all_scales > 0).sum() else 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale
        except FileNotFoundError:
            print(f"Error: depth_params.json not found at '{depth_params_file}'."); sys.exit(1)
        except Exception as e:
            print(f"Error opening depth_params.json: {e}"); sys.exit(1)

    # ---- Split train/test ----
    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images is None else images

    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir),
        depths_folder=os.path.join(path, depths) if depths != "" else "",
        test_cam_names_list=test_cam_names_list
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # ---- Prepare PLY ----
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3D.* to .ply (first time only)...")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)

    # ---- Load PCD ----
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    # ---- NEW: Filter PCD by multi-view masks, if masks/ exists ----
    masks_dir = os.path.join(path, "masks")
    if pcd is not None and os.path.isdir(masks_dir):
        print("[MaskFilter] masks/ found, filtering COLMAP point cloud...")
        # You can tune sample_cams and vis_thresh as needed (speed vs purity)
        pcd = filter_point_cloud_with_masks(
            pcd,
            cam_infos=train_cam_infos,     # use train cams for filtering
            masks_dir=masks_dir,
            sample_cams=20,                # check up to 20 views / point
            vis_thresh=0.30                # keep if >=30% of visible projections are foreground
        )
    else:
        if not os.path.isdir(masks_dir):
            print("[MaskFilter] masks/ not found, skip PCD filtering.")
        elif pcd is None:
            print("[MaskFilter] PCD not available, skip filtering.")

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

# =========================
# Blender readers (unchanged)
# =========================

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            # For Blender we don't have explicit fx,fy,cx,cy here; fill reasonable defaults
            fx = fov2focal(FovX, image.size[0]); fy = fov2focal(FovY, image.size[1])
            cx = image.size[0] * 0.5; cy = image.size[1] * 0.5

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path,
                            depth_params=None, is_test=is_test,
                            fx=fx, fy=fy, cx=cx, cy=cy))
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):
    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo
}
