"""
Vision-based PyBullet throw test for a trained MC-PILOT policy.

This script mirrors the style of test_mc_pilot_pb_A.py for configuration and
uses the existing PyBullet arm throw path from the project, but inserts a
perception stage before each throw:
  1. Spawn a visible target in PyBullet.
  2. Render RGB-D from a configurable camera.
  3. Run a local YOLO model on the RGB image.
  4. Back-project the detected target using depth + camera intrinsics/extrinsics.
  5. Feed the estimated target x-y position into the throwing policy.

Typical usage:
  python test_mc_pilot_pb_A_vision.py ^
      --log_path results_mc_pilot_pb_A/1 ^
      --yolo_model C:/models/target_detector.pt ^
      --target_class_name throw_target ^
      --num_throws 5
"""

import argparse
import csv
import json
import os
import pickle as pkl
import signal
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import pybullet as p
import pybullet_data
import torch

sys.path.insert(0, ".")
import policy_learning.Policy as Policy
from robot_arm.arm_controller import ArmController
from robot_arm.robot_profiles import available_robot_names, get_robot_profile
from simulation_class.model import _ball_accel
from simulation_class.model_pybullet import PyBulletThrowingSystem


BALL_MASS = 0.0577
BALL_RADIUS = 0.0327
LAUNCH_ANGLE_DEG = 35.0
STATE_DIM = 8
TARGET_DIM = 2
DTYPE = torch.float64
DEVICE = torch.device("cpu")


def find_local_model_candidates(search_root="."):
    candidates = []
    for root, _, files in os.walk(search_root):
        for name in files:
            if os.path.splitext(name)[1].lower() in {".pt", ".onnx", ".engine", ".torchscript"}:
                candidates.append(os.path.join(root, name))
    candidates.sort()
    return candidates


def prepare_ultralytics_env():
    config_dir = os.path.abspath(".ultralytics")
    os.makedirs(config_dir, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", config_dir)
    return config_dir


def build_arg_parser():
    parser = argparse.ArgumentParser("mc-pilot-pybullet vision-based throw test")
    parser.add_argument("--log_path", type=str, required=True, help="path to results/<seed>/ with log.pkl")
    parser.add_argument(
        "--yolo_model",
        type=str,
        required=True,
        help="local path to YOLO weights (.pt or equivalent) for target detection",
    )
    parser.add_argument(
        "--target_class_name",
        type=str,
        default=None,
        help="YOLO class name to keep, e.g. 'throw_target'; if unset, use highest-confidence detection",
    )
    parser.add_argument(
        "--target_class_id",
        type=int,
        default=None,
        help="YOLO class id to keep; ignored if --target_class_name resolves successfully",
    )
    parser.add_argument("--yolo_conf", type=float, default=0.25, help="minimum YOLO confidence threshold")
    parser.add_argument("--max_det", type=int, default=10, help="maximum detections returned by YOLO")
    parser.add_argument(
        "--robot",
        type=str,
        default=None,
        choices=available_robot_names(),
        help="robot arm profile to visualize; defaults to config value or kuka_iiwa",
    )
    parser.add_argument(
        "--use_profile_defaults",
        action="store_true",
        help="when overriding the robot, use that robot's safer release pose and speed cap",
    )
    parser.add_argument("--num_throws", type=int, default=5, help="number of perception-driven throws")
    parser.add_argument("--seed", type=int, default=42, help="seed for target sampling")
    parser.add_argument("--direct", action="store_true", help="run PyBullet headless")
    parser.add_argument("--slow", type=float, default=1.0, help="GUI playback slowdown factor")
    parser.add_argument("--record", type=str, default=None, help="optional output .mp4 path")
    parser.add_argument(
        "--record_mode",
        type=str,
        default="camera",
        choices=["camera", "overlay", "pybullet"],
        help="record raw camera frames, overlay frames, or PyBullet's built-in MP4 logger",
    )
    parser.add_argument("--record_fps", type=float, default=30.0, help="FPS for non-PyBullet MP4 recording")
    parser.add_argument(
        "--speed_scale",
        type=float,
        default=1.0,
        help="extra multiplier on the policy speed before clipping to uM",
    )
    parser.add_argument(
        "--speed_strategy",
        type=str,
        default="search",
        choices=["policy", "search", "hybrid"],
        help="how to choose release speed from the estimated target",
    )
    parser.add_argument(
        "--use_legacy_release_pos",
        action="store_true",
        help="keep the legacy release position from the training config instead of auto-switching to profile defaults",
    )
    parser.add_argument(
        "--search_grid_size",
        type=int,
        default=17,
        help="number of coarse speed samples for PyBullet search",
    )
    parser.add_argument(
        "--search_refine_steps",
        type=int,
        default=2,
        help="number of local refinement rounds after the coarse search",
    )
    parser.add_argument(
        "--search_refine_width",
        type=float,
        default=0.18,
        help="half-width in m/s of each local refinement window",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="directory for logs and optional debug artifacts",
    )
    parser.add_argument(
        "--fallback_to_gt_on_miss",
        action="store_true",
        help="if detection fails, use ground-truth target instead of skipping the throw",
    )
    parser.add_argument(
        "--draw_debug",
        action="store_true",
        help="draw camera rays, estimate markers, trajectory annotations, and a live RGB detection window",
    )
    parser.add_argument(
        "--keep_window_open",
        action="store_true",
        help="keep the GUI simulation alive after the last throw",
    )
    parser.add_argument(
        "--save_debug_frames",
        action="store_true",
        help="save per-throw RGB-D arrays and detection metadata to disk",
    )
    parser.add_argument(
        "--save_rgb_png",
        action="store_true",
        help="save RGB debug images with the chosen detection overlaid when Pillow is available",
    )
    parser.add_argument(
        "--save_depth_png",
        action="store_true",
        help="save depth visualizations when Pillow is available",
    )
    parser.add_argument("--hit_tolerance", type=float, default=0.10, help="hit radius in meters")
    parser.add_argument(
        "--depth_percentile",
        type=float,
        default=15.0,
        help="percentile of valid depth samples inside the central ROI of the box",
    )
    parser.add_argument(
        "--depth_roi_scale",
        type=float,
        default=0.40,
        help="fraction of bbox width/height used for the central depth ROI",
    )
    parser.add_argument(
        "--target_radius",
        type=float,
        default=0.06,
        help="rendered target sphere radius in meters; used to recover the sphere center from surface depth",
    )
    parser.add_argument(
        "--refine_red_target",
        action="store_true",
        default=True,
        help="refine the target center inside the YOLO box using the rendered red target pixels",
    )
    parser.add_argument(
        "--refine_with_segmentation",
        action="store_true",
        default=True,
        help="in PyBullet, refine the target center inside the YOLO box using the renderer segmentation mask",
    )
    parser.add_argument(
        "--red_min_ratio",
        type=float,
        default=1.35,
        help="minimum red dominance ratio used for target-pixel refinement inside the YOLO box",
    )
    parser.add_argument(
        "--camera_eye",
        type=float,
        nargs=3,
        default=[1.20, 0.00, 1.20],
        metavar=("X", "Y", "Z"),
        help="camera eye position in world coordinates",
    )
    parser.add_argument(
        "--camera_target",
        type=float,
        nargs=3,
        default=[0.85, 0.00, 0.05],
        metavar=("X", "Y", "Z"),
        help="camera look-at target in world coordinates",
    )
    parser.add_argument(
        "--camera_up",
        type=float,
        nargs=3,
        default=[0.00, 0.00, 1.00],
        metavar=("X", "Y", "Z"),
        help="camera up direction in world coordinates",
    )
    parser.add_argument("--camera_width", type=int, default=640, help="RGB-D image width")
    parser.add_argument("--camera_height", type=int, default=480, help="RGB-D image height")
    parser.add_argument("--camera_fov", type=float, default=60.0, help="vertical camera field of view in degrees")
    parser.add_argument("--camera_near", type=float, default=0.05, help="near clipping plane in meters")
    parser.add_argument("--camera_far", type=float, default=4.00, help="far clipping plane in meters")
    return parser


def sample_target(rng, lm, lM, gM):
    dist = rng.uniform(lm, lM)
    angle = rng.uniform(-gM, gM)
    return np.array([dist * np.cos(angle), dist * np.sin(angle)], dtype=float)


def make_output_dir(base_dir, seed):
    if base_dir is not None:
        out_dir = base_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("results_mc_pilot_pb_A_vision", f"seed_{seed}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def load_policy_and_config(log_path, robot_override=None, use_profile_defaults=False):
    log_file = os.path.join(log_path, "log.pkl")
    cfg_file = os.path.join(log_path, "config_log.pkl")
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"No log.pkl found at {log_file}")
    if not os.path.exists(cfg_file):
        raise FileNotFoundError(f"No config_log.pkl found at {cfg_file}")

    with open(log_file, "rb") as f:
        log = pkl.load(f)
    with open(cfg_file, "rb") as f:
        cfg = pkl.load(f)

    if "parameters_trial_list" not in log or not log["parameters_trial_list"]:
        raise ValueError(f"No trained policy weights found in {log_file}")

    cfg_robot_name = cfg.get("robot_name", "kuka_iiwa")
    robot_name = robot_override or cfg_robot_name
    profile = get_robot_profile(robot_name)
    robot_overridden = robot_name != cfg_robot_name
    use_profile_defaults = bool(use_profile_defaults or robot_overridden)

    lm = float(cfg.get("lm", 0.6))
    lM = float(cfg.get("lM", 1.1))
    gM = float(cfg.get("gM", np.pi / 6))
    uM = float(cfg.get("uM", profile.speed_bounds[1]))
    Ts = float(cfg.get("Ts", 0.02))
    T = float(cfg.get("T", 0.60))

    if use_profile_defaults:
        release_pos = np.array(profile.default_release_pos, dtype=float)
        uM = min(uM, float(profile.speed_bounds[1]))
    else:
        release_pos_cfg = cfg.get("release_pos")
        release_pos = np.array(
            release_pos_cfg if release_pos_cfg is not None else profile.default_release_pos,
            dtype=float,
        )

    policy_state = log["parameters_trial_list"][-1]
    Nb = int(cfg.get("Nb", 250))
    ls_init = np.asarray(cfg.get("lengthscales_init", [0.08, 0.08]), dtype=float)
    centers_init = policy_state["centers"].detach().cpu().numpy()
    weight_init = policy_state["f_linear.weight"].detach().cpu().numpy()
    ls_loaded = policy_state["log_lengthscales"].detach().exp().cpu().numpy()[0]

    policy_obj = Policy.Throwing_Policy(
        full_state_dim=STATE_DIM,
        target_dim=TARGET_DIM,
        num_basis=Nb,
        u_max=uM,
        lengthscales_init=ls_loaded if ls_loaded.shape[0] == TARGET_DIM else ls_init,
        centers_init=centers_init,
        weight_init=weight_init,
        flg_drop=False,
        dtype=DTYPE,
        device=DEVICE,
    )
    policy_obj.load_state_dict(policy_state)
    policy_obj.eval()

    runtime_cfg = {
        "cfg_robot_name": cfg_robot_name,
        "robot_name": robot_name,
        "robot_overridden": robot_overridden,
        "release_pos": release_pos,
        "lm": lm,
        "lM": lM,
        "gM": gM,
        "uM": uM,
        "Ts": Ts,
        "T": T,
        "Nb": Nb,
        "lengthscales_init": ls_init,
        "profile": profile,
    }
    return policy_obj, log, cfg, runtime_cfg


def get_speed(policy_obj, release_pos, target_xy):
    s0 = np.concatenate([release_pos, np.zeros(3), target_xy])
    with torch.no_grad():
        inp = torch.tensor(s0, dtype=DTYPE, device=DEVICE).unsqueeze(0)
        speed = policy_obj(inp, t=0, p_dropout=0.0)
    return float(speed.item())


def simulate_landing_xy(throwing_system, speed, release_pos, target_xy, flight_time, dt):
    s0 = np.concatenate([release_pos, np.zeros(3), target_xy])

    def fixed_speed_policy(_state, _t):
        return np.array([speed], dtype=float)

    clean_states, _, noiseless_states = None, None, None
    noisy_states, _, clean_states = throwing_system.rollout(
        s0=s0,
        policy=fixed_speed_policy,
        T=flight_time,
        dt=dt,
        noise=np.zeros(8, dtype=float),
    )
    landing_xy = clean_states[-1, 0:2].copy()
    return landing_xy


def choose_speed_for_target(
    speed_strategy,
    policy_obj,
    throwing_system,
    release_pos,
    target_xy,
    u_min,
    u_max,
    dt,
    flight_time,
    speed_scale,
    grid_size,
    refine_steps,
    refine_width,
):
    policy_speed = min(max(get_speed(policy_obj, release_pos, target_xy) * speed_scale, u_min), u_max)
    policy_landing_xy = simulate_landing_xy(
        throwing_system=throwing_system,
        speed=policy_speed,
        release_pos=release_pos,
        target_xy=target_xy,
        flight_time=flight_time,
        dt=dt,
    )
    policy_error = float(np.linalg.norm(policy_landing_xy - target_xy))

    if speed_strategy == "policy":
        return {
            "speed": float(policy_speed),
            "strategy_used": "policy",
            "policy_speed": float(policy_speed),
            "policy_error_pred_m": policy_error,
            "search_error_pred_m": None,
            "search_best_speed": None,
            "search_best_landing_xy": None,
            "policy_landing_xy": policy_landing_xy,
        }

    coarse_speeds = np.linspace(u_min, u_max, max(3, int(grid_size)))
    if speed_strategy == "hybrid":
        coarse_speeds = np.unique(np.concatenate([coarse_speeds, [policy_speed]])).astype(float)

    best_speed = None
    best_error = np.inf
    best_landing_xy = None

    for speed in coarse_speeds:
        landing_xy = simulate_landing_xy(
            throwing_system=throwing_system,
            speed=float(speed),
            release_pos=release_pos,
            target_xy=target_xy,
            flight_time=flight_time,
            dt=dt,
        )
        error = float(np.linalg.norm(landing_xy - target_xy))
        if error < best_error:
            best_error = error
            best_speed = float(speed)
            best_landing_xy = landing_xy

    for _ in range(max(0, int(refine_steps))):
        low = max(u_min, best_speed - refine_width)
        high = min(u_max, best_speed + refine_width)
        local_speeds = np.linspace(low, high, 9)
        for speed in local_speeds:
            landing_xy = simulate_landing_xy(
                throwing_system=throwing_system,
                speed=float(speed),
                release_pos=release_pos,
                target_xy=target_xy,
                flight_time=flight_time,
                dt=dt,
            )
            error = float(np.linalg.norm(landing_xy - target_xy))
            if error < best_error:
                best_error = error
                best_speed = float(speed)
                best_landing_xy = landing_xy

    use_policy = speed_strategy == "hybrid" and policy_error <= best_error
    if use_policy:
        return {
            "speed": float(policy_speed),
            "strategy_used": "policy",
            "policy_speed": float(policy_speed),
            "policy_error_pred_m": policy_error,
            "search_error_pred_m": best_error,
            "search_best_speed": float(best_speed),
            "search_best_landing_xy": best_landing_xy,
            "policy_landing_xy": policy_landing_xy,
        }

    return {
        "speed": float(best_speed),
        "strategy_used": "search",
        "policy_speed": float(policy_speed),
        "policy_error_pred_m": policy_error,
        "search_error_pred_m": float(best_error),
        "search_best_speed": float(best_speed),
        "search_best_landing_xy": best_landing_xy,
        "policy_landing_xy": policy_landing_xy,
    }


class YoloDetector:
    def __init__(self, model_path, conf_threshold, max_det, class_name=None, class_id=None):
        if not os.path.exists(model_path):
            local_candidates = find_local_model_candidates(".")
            candidate_lines = "\n".join(f"  - {path}" for path in local_candidates[:20])
            if local_candidates:
                candidate_msg = f"\nLocal model candidates found under mc-pilot-pybullet:\n{candidate_lines}"
            else:
                candidate_msg = "\nNo local YOLO-style weight files were found under mc-pilot-pybullet."
            raise FileNotFoundError(
                "YOLO weights not found.\n"
                f"Passed --yolo_model: {model_path}\n"
                "That example path is only a placeholder. Pass the real local path to your detector weights.\n"
                "Example:\n"
                "  python test_mc_pilot_pb_A_vision.py --log_path results_mc_pilot_pb_A/1 "
                "--yolo_model C:/real/path/to/weights.pt --target_class_name \"throw_target\"\n"
                f"{candidate_msg}"
            )
        prepare_ultralytics_env()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for YOLO inference. Install it in the local environment "
                "and provide weights via --yolo_model."
            ) from exc

        self.model = YOLO(model_path)
        self.conf_threshold = float(conf_threshold)
        self.max_det = int(max_det)
        self.class_name = class_name
        self.class_id = class_id

    def predict(self, rgb_image):
        # Ultralytics expects images in OpenCV/PIL style; BGR array works reliably.
        bgr_image = np.ascontiguousarray(rgb_image[:, :, ::-1])
        results = self.model.predict(
            source=bgr_image,
            conf=self.conf_threshold,
            max_det=self.max_det,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        raw_names = result.names if hasattr(result, "names") else {}
        if isinstance(raw_names, dict):
            names = dict(raw_names)
        else:
            names = {idx: name for idx, name in enumerate(raw_names)}

        target_class_id = self.class_id
        if self.class_name is not None:
            matches = [int(k) for k, v in names.items() if v == self.class_name]
            if not matches:
                available = ", ".join(str(v) for _, v in sorted(names.items()))
                raise ValueError(
                    f"Class name '{self.class_name}' not present in model classes. Available: {available}"
                )
            target_class_id = matches[0]

        detections = []
        if boxes is not None:
            for idx in range(len(boxes)):
                conf = float(boxes.conf[idx].item())
                cls_id = int(boxes.cls[idx].item())
                if target_class_id is not None and cls_id != target_class_id:
                    continue
                xyxy = boxes.xyxy[idx].detach().cpu().numpy().astype(float)
                detections.append(
                    {
                        "bbox_xyxy": xyxy,
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": names.get(cls_id, str(cls_id)),
                    }
                )

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        best = detections[0] if detections else None
        return best, detections


def build_camera_config(args):
    width = int(args.camera_width)
    height = int(args.camera_height)
    aspect = float(width) / float(height)
    fov_y = np.deg2rad(float(args.camera_fov))
    tan_half_fov_y = np.tan(fov_y / 2.0)
    tan_half_fov_x = aspect * tan_half_fov_y
    fx = width / (2.0 * tan_half_fov_x)
    fy = height / (2.0 * tan_half_fov_y)
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=list(args.camera_eye),
        cameraTargetPosition=list(args.camera_target),
        cameraUpVector=list(args.camera_up),
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=float(args.camera_fov),
        aspect=aspect,
        nearVal=float(args.camera_near),
        farVal=float(args.camera_far),
    )

    view_mat = np.array(view_matrix, dtype=float).reshape(4, 4, order="F")
    proj_mat = np.array(projection_matrix, dtype=float).reshape(4, 4, order="F")
    cam_to_world = np.linalg.inv(view_mat)

    return {
        "width": width,
        "height": height,
        "aspect": aspect,
        "fov_deg": float(args.camera_fov),
        "near": float(args.camera_near),
        "far": float(args.camera_far),
        "eye": np.asarray(args.camera_eye, dtype=float),
        "target": np.asarray(args.camera_target, dtype=float),
        "up": np.asarray(args.camera_up, dtype=float),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "view_matrix": view_mat,
        "projection_matrix": proj_mat,
        "cam_to_world": cam_to_world,
        "world_to_cam": view_mat,
        "view_matrix_flat": np.asarray(view_matrix, dtype=float),
        "projection_matrix_flat": np.asarray(projection_matrix, dtype=float),
    }


def render_rgbd(client_id, camera_cfg, direct_mode):
    renderer = p.ER_TINY_RENDERER if direct_mode else p.ER_BULLET_HARDWARE_OPENGL
    _, _, rgba, depth_buffer, seg = p.getCameraImage(
        width=camera_cfg["width"],
        height=camera_cfg["height"],
        viewMatrix=camera_cfg["view_matrix_flat"].tolist(),
        projectionMatrix=camera_cfg["projection_matrix_flat"].tolist(),
        renderer=renderer,
        physicsClientId=client_id,
    )
    rgba = np.reshape(rgba, (camera_cfg["height"], camera_cfg["width"], 4))
    rgb = rgba[:, :, :3].astype(np.uint8)
    depth_buffer = np.reshape(depth_buffer, (camera_cfg["height"], camera_cfg["width"])).astype(np.float64)
    seg = np.reshape(seg, (camera_cfg["height"], camera_cfg["width"]))

    near = camera_cfg["near"]
    far = camera_cfg["far"]
    depth_m = far * near / (far - (far - near) * depth_buffer)
    depth_m = depth_m.astype(np.float64)
    return rgb, depth_m, depth_buffer, seg


def clip_bbox_xyxy(bbox_xyxy, width, height):
    x1 = int(np.clip(np.floor(bbox_xyxy[0]), 0, width - 1))
    y1 = int(np.clip(np.floor(bbox_xyxy[1]), 0, height - 1))
    x2 = int(np.clip(np.ceil(bbox_xyxy[2]), 0, width - 1))
    y2 = int(np.clip(np.ceil(bbox_xyxy[3]), 0, height - 1))
    return x1, y1, x2, y2


def select_depth_from_bbox(depth_m, bbox_xyxy, roi_scale, percentile):
    height, width = depth_m.shape
    x1, y1, x2, y2 = clip_bbox_xyxy(bbox_xyxy, width, height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox after clipping: {(x1, y1, x2, y2)}")

    u_center = 0.5 * (x1 + x2)
    v_center = 0.5 * (y1 + y2)

    roi_w = max(1, int(round((x2 - x1 + 1) * roi_scale)))
    roi_h = max(1, int(round((y2 - y1 + 1) * roi_scale)))
    u_mid = int(round(u_center))
    v_mid = int(round(v_center))
    rx1 = max(0, u_mid - roi_w // 2)
    ry1 = max(0, v_mid - roi_h // 2)
    rx2 = min(width - 1, rx1 + roi_w - 1)
    ry2 = min(height - 1, ry1 + roi_h - 1)

    roi = depth_m[ry1 : ry2 + 1, rx1 : rx2 + 1]
    valid = roi[np.isfinite(roi) & (roi > 0.0)]
    if valid.size == 0:
        raise ValueError("No valid metric depth values found inside the detection ROI.")

    surface_depth = float(np.percentile(valid, percentile))
    return {
        "bbox_px": [x1, y1, x2, y2],
        "roi_px": [rx1, ry1, rx2, ry2],
        "pixel_uv": [float(u_center), float(v_center)],
        "surface_depth_m": surface_depth,
        "num_valid_depth_samples": int(valid.size),
    }


def refine_target_pixels(rgb_image, depth_m, bbox_xyxy, red_min_ratio):
    height, width, _ = rgb_image.shape
    x1, y1, x2, y2 = clip_bbox_xyxy(bbox_xyxy, width, height)
    if x2 <= x1 or y2 <= y1:
        return None

    patch = rgb_image[y1 : y2 + 1, x1 : x2 + 1].astype(np.float32)
    depth_patch = depth_m[y1 : y2 + 1, x1 : x2 + 1]
    red = patch[:, :, 0]
    green = patch[:, :, 1]
    blue = patch[:, :, 2]
    max_gb = np.maximum(green, blue)
    valid_depth = np.isfinite(depth_patch) & (depth_patch > 0.0)

    red_mask = (
        valid_depth
        & (red >= 80.0)
        & (red / np.maximum(max_gb, 1.0) >= red_min_ratio)
        & ((red - 0.5 * (green + blue)) >= 30.0)
    )
    if int(red_mask.sum()) < 12:
        return None

    ys, xs = np.where(red_mask)
    u = float(x1 + xs.mean())
    v = float(y1 + ys.mean())
    surface_depth = float(np.median(depth_patch[red_mask]))
    rx1 = int(x1 + xs.min())
    ry1 = int(y1 + ys.min())
    rx2 = int(x1 + xs.max())
    ry2 = int(y1 + ys.max())
    return {
        "bbox_px": [x1, y1, x2, y2],
        "roi_px": [rx1, ry1, rx2, ry2],
        "pixel_uv": [u, v],
        "surface_depth_m": surface_depth,
        "num_valid_depth_samples": int(np.count_nonzero(valid_depth)),
        "num_refined_pixels": int(red_mask.sum()),
        "refinement_mode": "red-pixel-mask",
    }


def refine_target_segmentation(seg_image, depth_m, bbox_xyxy, target_body_id):
    if target_body_id is None:
        return None
    height, width = depth_m.shape
    x1, y1, x2, y2 = clip_bbox_xyxy(bbox_xyxy, width, height)
    if x2 <= x1 or y2 <= y1:
        return None

    seg_patch = seg_image[y1 : y2 + 1, x1 : x2 + 1]
    depth_patch = depth_m[y1 : y2 + 1, x1 : x2 + 1]
    mask = (seg_patch == int(target_body_id)) & np.isfinite(depth_patch) & (depth_patch > 0.0)
    if int(mask.sum()) < 8:
        return None

    ys, xs = np.where(mask)
    u = float(x1 + xs.mean())
    v = float(y1 + ys.mean())
    surface_depth = float(np.median(depth_patch[mask]))
    rx1 = int(x1 + xs.min())
    ry1 = int(y1 + ys.min())
    rx2 = int(x1 + xs.max())
    ry2 = int(y1 + ys.max())
    return {
        "bbox_px": [x1, y1, x2, y2],
        "roi_px": [rx1, ry1, rx2, ry2],
        "pixel_uv": [u, v],
        "surface_depth_m": surface_depth,
        "num_valid_depth_samples": int(mask.sum()),
        "num_refined_pixels": int(mask.sum()),
        "refinement_mode": "segmentation-mask",
    }


def backproject_pixel_to_world(u, v, depth_m, camera_cfg):
    x_cam = (u - camera_cfg["cx"]) * depth_m / camera_cfg["fx"]
    y_cam = -(v - camera_cfg["cy"]) * depth_m / camera_cfg["fy"]
    z_cam = -depth_m
    point_cam_h = np.array([x_cam, y_cam, z_cam, 1.0], dtype=float)
    point_world_h = camera_cfg["cam_to_world"] @ point_cam_h
    return point_world_h[:3] / point_world_h[3]


def unproject_with_view_projection(u, v, depth_buffer_value, camera_cfg):
    x_ndc = 2.0 * ((u + 0.5) / camera_cfg["width"]) - 1.0
    y_ndc = 1.0 - 2.0 * ((v + 0.5) / camera_cfg["height"])
    z_ndc = 2.0 * depth_buffer_value - 1.0
    clip = np.array([x_ndc, y_ndc, z_ndc, 1.0], dtype=float)
    inv_pv = np.linalg.inv(camera_cfg["projection_matrix"] @ camera_cfg["view_matrix"])
    world_h = inv_pv @ clip
    return world_h[:3] / world_h[3]


def estimate_target_world(
    detection,
    rgb_image,
    depth_m,
    depth_buffer,
    seg_image,
    camera_cfg,
    target_radius,
    roi_scale,
    percentile,
    refine_with_segmentation,
    refine_red_target,
    red_min_ratio,
    target_body_id,
):
    if refine_with_segmentation:
        depth_info = refine_target_segmentation(
            seg_image=seg_image,
            depth_m=depth_m,
            bbox_xyxy=detection["bbox_xyxy"],
            target_body_id=target_body_id,
        )
    else:
        depth_info = None

    if depth_info is None and refine_red_target:
        depth_info = refine_target_pixels(
            rgb_image=rgb_image,
            depth_m=depth_m,
            bbox_xyxy=detection["bbox_xyxy"],
            red_min_ratio=red_min_ratio,
        )
    else:
        depth_info = None

    if depth_info is None:
        depth_info = select_depth_from_bbox(
            depth_m=depth_m,
            bbox_xyxy=detection["bbox_xyxy"],
            roi_scale=roi_scale,
            percentile=percentile,
        )
        depth_info["refinement_mode"] = "bbox-roi"

    u, v = depth_info["pixel_uv"]
    surface_world = backproject_pixel_to_world(u, v, depth_info["surface_depth_m"], camera_cfg)

    u_px = int(np.clip(round(u), 0, camera_cfg["width"] - 1))
    v_px = int(np.clip(round(v), 0, camera_cfg["height"] - 1))
    vp_world = unproject_with_view_projection(u_px, v_px, depth_buffer[v_px, u_px], camera_cfg)

    cam_origin = camera_cfg["cam_to_world"][:3, 3]
    ray_dir = surface_world - cam_origin
    ray_norm = np.linalg.norm(ray_dir)
    if ray_norm <= 1e-9:
        raise ValueError("Camera ray for target back-projection has near-zero norm.")
    ray_dir = ray_dir / ray_norm

    center_world = surface_world + target_radius * ray_dir
    depth_info["surface_world_xyz"] = surface_world
    depth_info["center_world_xyz"] = center_world
    depth_info["vp_world_xyz"] = vp_world
    depth_info["surface_vs_vp_error_m"] = float(np.linalg.norm(surface_world - vp_world))
    return depth_info


def draw_camera_debug(client_id, camera_cfg):
    eye = camera_cfg["eye"]
    target = camera_cfg["target"]
    p.addUserDebugLine(eye.tolist(), target.tolist(), [1, 1, 1], 1.5, physicsClientId=client_id)
    p.addUserDebugText(
        "camera",
        eye.tolist(),
        textColorRGB=[1, 1, 1],
        textSize=1.2,
        physicsClientId=client_id,
    )


def maybe_save_debug_images(base_path, rgb_image, depth_m, chosen_detection, depth_info, save_rgb, save_depth):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("  Pillow not available; skipping PNG debug image export.")
        return

    if save_rgb:
        rgb_im = Image.fromarray(rgb_image)
        draw = ImageDraw.Draw(rgb_im)
        if chosen_detection is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in chosen_detection["bbox_xyxy"]]
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
            if depth_info is not None:
                u, v = depth_info["pixel_uv"]
                r = 4
                draw.ellipse([u - r, v - r, u + r, v + r], fill=(255, 255, 0))
                roi = depth_info["roi_px"]
                draw.rectangle(roi, outline=(255, 255, 0), width=2)
            draw.text(
                (x1 + 4, max(0, y1 - 16)),
                f"{chosen_detection['class_name']} {chosen_detection['confidence']:.2f}",
                fill=(255, 255, 255),
            )
        rgb_im.save(base_path + "_rgb.png")

    if save_depth:
        depth_min = float(np.nanmin(depth_m))
        depth_max = float(np.nanmax(depth_m))
        denom = max(depth_max - depth_min, 1e-6)
        depth_vis = 255.0 * (1.0 - (depth_m - depth_min) / denom)
        depth_vis = np.clip(depth_vis, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(depth_vis, mode="L").save(base_path + "_depth.png")


def make_detection_overlay(rgb_image, chosen_detection, all_detections, depth_info, gt_target_xy, est_target_world):
    overlay_bgr = np.ascontiguousarray(rgb_image[:, :, ::-1].copy())

    for idx, det in enumerate(all_detections):
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy"]]
        color = (0, 255, 0) if idx == 0 and chosen_detection is not None and det is chosen_detection else (0, 180, 255)
        cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.putText(
            overlay_bgr,
            label,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    info_lines = [f"GT xy: ({gt_target_xy[0]:.3f}, {gt_target_xy[1]:.3f})"]
    if chosen_detection is not None and depth_info is not None and est_target_world is not None:
        u, v = depth_info["pixel_uv"]
        cv2.circle(overlay_bgr, (int(round(u)), int(round(v))), 4, (0, 255, 255), -1)
        rx1, ry1, rx2, ry2 = [int(v) for v in depth_info["roi_px"]]
        cv2.rectangle(overlay_bgr, (rx1, ry1), (rx2, ry2), (0, 255, 255), 1)
        info_lines.append(f"Est xy: ({est_target_world[0]:.3f}, {est_target_world[1]:.3f})")
        info_lines.append(f"Depth: {depth_info['surface_depth_m']:.3f} m")
        info_lines.append(f"Perception err: {np.linalg.norm(est_target_world[:2] - gt_target_xy):.3f} m")
    else:
        info_lines.append("Est xy: unavailable")

    y0 = 24
    for idx, line in enumerate(info_lines):
        cv2.putText(
            overlay_bgr,
            line,
            (10, y0 + 24 * idx),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay_bgr


def show_detection_overlay(window_name, overlay_bgr, wait_ms=1):
    cv2.imshow(window_name, overlay_bgr)
    cv2.waitKey(wait_ms)


class VideoRecorder:
    def __init__(self, output_path, width, height, fps):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(output_path, fourcc, float(fps), (int(width), int(height)))
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")
        self.output_path = output_path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)

    def write_bgr(self, frame_bgr):
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
        self._writer.write(np.ascontiguousarray(frame_bgr))

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def make_camera_frame_bgr(client_id, camera_cfg, direct_mode):
    rgb_image, _, _, _ = render_rgbd(client_id, camera_cfg, direct_mode)
    return np.ascontiguousarray(rgb_image[:, :, ::-1])


def create_target_visual(client_id, target_xy, target_radius, rgba):
    target_pos = np.array([target_xy[0], target_xy[1], target_radius], dtype=float)
    target_vis = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=target_radius,
        rgbaColor=rgba,
        physicsClientId=client_id,
    )
    target_id = p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=target_vis,
        basePosition=target_pos.tolist(),
        physicsClientId=client_id,
    )
    return target_id, target_pos


def execute_throw(
    client_id,
    arm,
    policy_target_xy,
    gt_target_xy,
    release_pos,
    speed,
    dt,
    flight_time,
    hit_tolerance,
    target_radius,
    direct_mode,
    slow_factor,
    draw_debug,
    camera_cfg=None,
    frame_callback=None,
  ):
    launch_angle = np.deg2rad(LAUNCH_ANGLE_DEG)
    T_W, T_R, T_ARM = get_robot_profile(arm.robot_name).timing

    dx = policy_target_xy[0] - release_pos[0]
    dy = policy_target_xy[1] - release_pos[1]
    phi = np.arctan2(dy, dx)
    v_cmd = np.array(
        [
            speed * np.cos(launch_angle) * np.cos(phi),
            speed * np.cos(launch_angle) * np.sin(phi),
            speed * np.sin(launch_angle),
        ],
        dtype=float,
    )

    arm.reset()
    p.stepSimulation(physicsClientId=client_id)
    if frame_callback is not None and camera_cfg is not None:
        frame_callback(make_camera_frame_bgr(client_id, camera_cfg, direct_mode))
    if not direct_mode:
        time.sleep(0.30 * slow_factor)

    ee_pos, _, _, _ = arm.ee_state()
    ball_col = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_RADIUS, physicsClientId=client_id)
    ball_vis = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=BALL_RADIUS,
        rgbaColor=[1, 1, 0, 1],
        physicsClientId=client_id,
    )
    ball_id = p.createMultiBody(
        baseMass=BALL_MASS,
        baseCollisionShapeIndex=ball_col,
        baseVisualShapeIndex=ball_vis,
        basePosition=ee_pos.tolist(),
        physicsClientId=client_id,
    )
    p.changeDynamics(ball_id, -1, linearDamping=0.0, angularDamping=0.0, physicsClientId=client_id)

    arm.attach_ball(ball_id)
    coeffs, q_release, qd_release, v_achieved = arm.plan_throw(v_cmd, release_pos, T_W, T_R, T_ARM)

    released = False
    ball_positions = []
    n_steps = int(T_ARM / dt) + int(flight_time / dt) + 50
    release_vel = None
    release_time = None

    for step in range(n_steps):
        t = step * dt
        if not released:
            q_t, qd_t = arm.get_setpoint(coeffs, t)
            arm.step(q_t, qd_t)
            if t >= T_R:
                release_vel = arm.release_ball(ball_id, set_vel=v_cmd)
                release_time = t
                released = True
        else:
            ball_pos_raw, _ = p.getBasePositionAndOrientation(ball_id, physicsClientId=client_id)
            ball_vel_raw, _ = p.getBaseVelocity(ball_id, physicsClientId=client_id)
            pos = np.array(ball_pos_raw, dtype=float)
            vel = np.array(ball_vel_raw, dtype=float)

            a_total = _ball_accel(pos, vel, mass=BALL_MASS, radius=BALL_RADIUS, wind=np.zeros(3))
            a_drag = a_total - np.array([0.0, 0.0, -9.81], dtype=float)
            f_drag = BALL_MASS * a_drag
            p.applyExternalForce(
                ball_id,
                -1,
                f_drag.tolist(),
                [0, 0, 0],
                p.WORLD_FRAME,
                physicsClientId=client_id,
            )

            ball_positions.append(pos.copy())
            if draw_debug and len(ball_positions) > 1:
                p.addUserDebugLine(
                    ball_positions[-2].tolist(),
                    ball_positions[-1].tolist(),
                    lineColorRGB=[0.0, 0.4, 1.0],
                    lineWidth=2,
                    physicsClientId=client_id,
                )

            if pos[2] <= BALL_RADIUS + 0.005 and len(ball_positions) > 3:
                break

        p.stepSimulation(physicsClientId=client_id)
        if frame_callback is not None and camera_cfg is not None:
            frame_callback(make_camera_frame_bgr(client_id, camera_cfg, direct_mode))
        if not direct_mode:
            time.sleep(dt * slow_factor)

    landing = None
    gt_error = None
    est_error = None
    hit_gt = False
    hit_est = False
    if ball_positions:
        landing = ball_positions[-1]
        gt_error = float(np.linalg.norm(landing[:2] - gt_target_xy))
        est_error = float(np.linalg.norm(landing[:2] - policy_target_xy))
        hit_gt = gt_error < hit_tolerance
        hit_est = est_error < hit_tolerance

        if draw_debug:
            color = [0, 1, 0, 0.9] if hit_gt else [1, 0.5, 0, 0.9]
            land_vis = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.04,
                rgbaColor=color,
                physicsClientId=client_id,
            )
            p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=land_vis,
                basePosition=landing.tolist(),
                physicsClientId=client_id,
            )
            p.addUserDebugText(
                f"gt err {gt_error * 100.0:.1f} cm",
                (landing + np.array([0.0, 0.0, 0.15])).tolist(),
                textColorRGB=color[:3],
                textSize=1.3,
                physicsClientId=client_id,
            )

    return {
        "ball_id": ball_id,
        "v_cmd": v_cmd,
        "q_release": q_release,
        "qd_release": qd_release,
        "v_achieved": v_achieved,
        "release_vel": release_vel,
        "release_time": release_time,
        "landing": landing,
        "landing_error_gt_m": gt_error,
        "landing_error_est_m": est_error,
        "hit_gt": hit_gt,
        "hit_est": hit_est,
    }


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if hasattr(value, "__dict__") and hasattr(value, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in value.__dict__.items()}
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_run_config(output_dir, args, train_cfg, runtime_cfg, camera_cfg):
    payload = {
        "args": vars(args),
        "train_cfg": to_jsonable(train_cfg),
        "runtime_cfg": to_jsonable(runtime_cfg),
        "camera_cfg": to_jsonable(
            {
                "width": camera_cfg["width"],
                "height": camera_cfg["height"],
                "fov_deg": camera_cfg["fov_deg"],
                "near": camera_cfg["near"],
                "far": camera_cfg["far"],
                "eye": camera_cfg["eye"],
                "target": camera_cfg["target"],
                "up": camera_cfg["up"],
                "fx": camera_cfg["fx"],
                "fy": camera_cfg["fy"],
                "cx": camera_cfg["cx"],
                "cy": camera_cfg["cy"],
                "view_matrix": camera_cfg["view_matrix"],
                "projection_matrix": camera_cfg["projection_matrix"],
                "cam_to_world": camera_cfg["cam_to_world"],
            }
        ),
    }
    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)

    policy_obj, _, train_cfg, runtime_cfg = load_policy_and_config(
        log_path=args.log_path,
        robot_override=args.robot,
        use_profile_defaults=args.use_profile_defaults,
    )
    profile = runtime_cfg["profile"]
    release_pos = runtime_cfg["release_pos"]
    Ts = runtime_cfg["Ts"]
    T = runtime_cfg["T"]
    lm = runtime_cfg["lm"]
    lM = runtime_cfg["lM"]
    gM = runtime_cfg["gM"]
    uM = runtime_cfg["uM"]
    search_u_min = float(train_cfg.get("uMin", profile.speed_bounds[0]))

    if (
        not args.use_profile_defaults
        and not args.use_legacy_release_pos
        and profile.name == "kuka_iiwa"
        and np.linalg.norm(release_pos[:2]) < 0.10
    ):
        release_pos = np.array(profile.default_release_pos, dtype=float)
        runtime_cfg["release_pos"] = release_pos.copy()
        uM = min(float(uM), float(profile.speed_bounds[1]))
        runtime_cfg["uM"] = uM
        search_u_min = float(profile.speed_bounds[0])

    output_dir = make_output_dir(args.output_dir, args.seed)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    camera_cfg = build_camera_config(args)
    write_run_config(output_dir, args, train_cfg, runtime_cfg, camera_cfg)

    detector = YoloDetector(
        model_path=args.yolo_model,
        conf_threshold=args.yolo_conf,
        max_det=args.max_det,
        class_name=args.target_class_name,
        class_id=args.target_class_id,
    )

    csv_path = os.path.join(output_dir, "throw_log.csv")
    csv_fields = [
        "throw_idx",
        "gt_target_x",
        "gt_target_y",
        "gt_target_z",
        "detection_found",
        "detection_confidence",
        "detection_class_name",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "depth_surface_m",
        "est_target_x",
        "est_target_y",
        "est_target_z",
        "perception_error_m",
        "surface_vs_vp_error_m",
        "used_target_x",
        "used_target_y",
        "used_ground_truth_fallback",
        "speed_strategy_requested",
        "speed_strategy_used",
        "policy_speed_mps",
        "policy_predicted_error_m",
        "search_best_speed_mps",
        "search_predicted_error_m",
        "planned_vx",
        "planned_vy",
        "planned_vz",
        "achieved_speed_mps",
        "release_time_s",
        "landing_x",
        "landing_y",
        "landing_z",
        "landing_error_gt_m",
        "landing_error_est_m",
        "hit_gt",
        "hit_est",
        "skipped_throw",
        "skip_reason",
    ]

    mode = p.DIRECT if args.direct else p.GUI
    client = p.connect(mode)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(Ts, physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.loadURDF("plane.urdf", physicsClientId=client)
    if not args.direct:
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[
                float(release_pos[0]),
                float(release_pos[1]),
                max(0.25, float(release_pos[2]) * 0.7),
            ],
            physicsClientId=client,
        )
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=client)

    urdf_path = pybullet_data.getDataPath() + "/" + profile.urdf_rel_path
    arm = ArmController(client, urdf_path, robot_name=profile.name)
    search_throwing_system = PyBulletThrowingSystem(
        mass=BALL_MASS,
        radius=BALL_RADIUS,
        launch_angle_deg=LAUNCH_ANGLE_DEG,
        arm_noise=None,
        gui_mode=False,
        robot_name=profile.name,
    )

    log_id = None
    if args.record is not None and args.record_mode == "pybullet":
        log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, args.record, physicsClientId=client)

    print(f"Loaded policy from {args.log_path}")
    print(f"Robot profile: {profile.name}")
    if runtime_cfg["robot_overridden"]:
        print(f"Config robot={runtime_cfg['cfg_robot_name']} but running robot={profile.name}")
    print(f"Release position: {np.round(release_pos, 4)}")
    if profile.name == "kuka_iiwa" and not args.use_legacy_release_pos and np.allclose(
        release_pos, np.array(profile.default_release_pos, dtype=float)
    ):
        print("Using the profile-default KUKA release pose for stable throws in the vision test.")
    print(f"Target range: lm={lm:.3f}, lM={lM:.3f}, gM={gM:.3f} rad")
    print(f"Speed search bounds: [{search_u_min:.3f}, {uM:.3f}] m/s")
    print(
        f"Rendered target object: red sphere marker with radius={args.target_radius:.3f} m "
        "placed at the sampled target position"
    )
    print(f"Camera eye={np.round(camera_cfg['eye'], 3)} target={np.round(camera_cfg['target'], 3)}")
    print(f"Logging to {output_dir}")

    rows = []
    debug_window_name = "MC-PILOT Vision Detection"
    video_recorder = None
    stop_requested = {"value": False}

    def _request_stop(_signum=None, _frame=None):
        stop_requested["value"] = True

    old_sigint = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, _request_stop)
    except Exception:
        old_sigint = None
    try:
        if args.record is not None and args.record_mode != "pybullet":
            video_recorder = VideoRecorder(
                output_path=args.record,
                width=camera_cfg["width"],
                height=camera_cfg["height"],
                fps=args.record_fps,
            )
        for throw_idx in range(args.num_throws):
            if stop_requested["value"]:
                print("Stop requested. Finishing cleanly and closing the video file.")
                break
            p.removeAllUserDebugItems(physicsClientId=client)
            if args.draw_debug:
                draw_camera_debug(client, camera_cfg)

            gt_target_xy = sample_target(rng, lm, lM, gM)
            target_id, gt_target_world = create_target_visual(
                client_id=client,
                target_xy=gt_target_xy,
                target_radius=args.target_radius,
                rgba=[1.0, 0.0, 0.0, 0.85],
            )

            if args.draw_debug:
                p.addUserDebugText(
                    f"GT ({gt_target_xy[0]:.2f}, {gt_target_xy[1]:.2f})",
                    (gt_target_world + np.array([0.0, 0.0, 0.12])).tolist(),
                    textColorRGB=[1.0, 0.3, 0.3],
                    textSize=1.2,
                    physicsClientId=client,
                )

            rgb_image, depth_m, depth_buffer, seg_image = render_rgbd(client, camera_cfg, args.direct)
            chosen_detection, all_detections = detector.predict(rgb_image)

            used_gt_fallback = False
            skipped_throw = False
            skip_reason = ""
            depth_info = None
            est_target_world = None
            est_target_xy = None

            if chosen_detection is not None:
                try:
                    depth_info = estimate_target_world(
                        detection=chosen_detection,
                        rgb_image=rgb_image,
                        depth_m=depth_m,
                        depth_buffer=depth_buffer,
                        seg_image=seg_image,
                        camera_cfg=camera_cfg,
                        target_radius=args.target_radius,
                        roi_scale=args.depth_roi_scale,
                        percentile=args.depth_percentile,
                        refine_with_segmentation=args.refine_with_segmentation,
                        refine_red_target=args.refine_red_target,
                        red_min_ratio=args.red_min_ratio,
                        target_body_id=target_id,
                    )
                    est_target_world = depth_info["center_world_xyz"]
                    est_target_xy = est_target_world[:2].copy()
                except Exception as exc:
                    skip_reason = f"depth-backprojection failed: {exc}"
                    chosen_detection = None

            if chosen_detection is None:
                if args.fallback_to_gt_on_miss:
                    used_target_xy = gt_target_xy.copy()
                    used_gt_fallback = True
                else:
                    used_target_xy = None
                    skipped_throw = True
                    if not skip_reason:
                        skip_reason = "no valid YOLO detection"
            else:
                used_target_xy = est_target_xy.copy()

            print(f"\nThrow {throw_idx + 1}/{args.num_throws}")
            print(f"  GT target xy: {np.round(gt_target_xy, 4)}")
            if chosen_detection is not None:
                print(
                    f"  Detection: {chosen_detection['class_name']} conf={chosen_detection['confidence']:.3f} "
                    f"bbox={np.round(chosen_detection['bbox_xyxy'], 1)}"
                )
                print(f"  Estimated target xyz: {np.round(est_target_world, 4)}")
                print(
                    f"  Perception xy error: {np.linalg.norm(est_target_world[:2] - gt_target_xy):.4f} m "
                    f"(surface-vp check {depth_info['surface_vs_vp_error_m']:.4f} m, "
                    f"mode={depth_info.get('refinement_mode', 'unknown')})"
                )
            else:
                print(f"  Detection: none ({skip_reason})")

            if args.draw_debug:
                overlay_bgr = make_detection_overlay(
                    rgb_image=rgb_image,
                    chosen_detection=chosen_detection,
                    all_detections=all_detections,
                    depth_info=depth_info,
                    gt_target_xy=gt_target_xy,
                    est_target_world=est_target_world,
                )
                show_detection_overlay(debug_window_name, overlay_bgr, wait_ms=1)
                if video_recorder is not None and args.record_mode == "overlay":
                    for _ in range(max(1, int(round(args.record_fps * 0.5)))):
                        video_recorder.write_bgr(overlay_bgr)
            elif video_recorder is not None and args.record_mode == "camera":
                video_recorder.write_bgr(np.ascontiguousarray(rgb_image[:, :, ::-1]))

            if args.save_debug_frames:
                frame_base = os.path.join(frames_dir, f"throw_{throw_idx + 1:03d}")
                np.savez_compressed(
                    frame_base + ".npz",
                    rgb_image=rgb_image,
                    depth_m=depth_m,
                    depth_buffer=depth_buffer,
                    gt_target_world=gt_target_world,
                    gt_target_xy=gt_target_xy,
                    chosen_detection_bbox=None if chosen_detection is None else chosen_detection["bbox_xyxy"],
                    chosen_detection_confidence=None if chosen_detection is None else chosen_detection["confidence"],
                    all_detections=np.array([json.dumps(to_jsonable(det)) for det in all_detections], dtype=object),
                    depth_info_json=json.dumps(to_jsonable(depth_info)),
                    camera_eye=camera_cfg["eye"],
                    camera_target=camera_cfg["target"],
                    camera_up=camera_cfg["up"],
                    view_matrix=camera_cfg["view_matrix"],
                    projection_matrix=camera_cfg["projection_matrix"],
                    cam_to_world=camera_cfg["cam_to_world"],
                )
                if args.save_rgb_png or args.save_depth_png:
                    maybe_save_debug_images(
                        frame_base,
                        rgb_image,
                        depth_m,
                        chosen_detection,
                        depth_info,
                        save_rgb=args.save_rgb_png,
                        save_depth=args.save_depth_png,
                    )

            throw_result = None
            speed = None
            if not skipped_throw:
                if args.draw_debug and chosen_detection is not None:
                    p.addUserDebugLine(
                        camera_cfg["eye"].tolist(),
                        est_target_world.tolist(),
                        [0.0, 1.0, 0.0],
                        1.5,
                        physicsClientId=client,
                    )
                    est_vis = p.createVisualShape(
                        p.GEOM_SPHERE,
                        radius=max(0.02, 0.35 * args.target_radius),
                        rgbaColor=[0.0, 1.0, 0.0, 0.85],
                        physicsClientId=client,
                    )
                    p.createMultiBody(
                        baseMass=0.0,
                        baseVisualShapeIndex=est_vis,
                        basePosition=est_target_world.tolist(),
                        physicsClientId=client,
                    )

                speed_choice = choose_speed_for_target(
                    speed_strategy=args.speed_strategy,
                    policy_obj=policy_obj,
                    throwing_system=search_throwing_system,
                    release_pos=release_pos,
                    target_xy=used_target_xy,
                    u_min=search_u_min,
                    u_max=float(uM),
                    dt=Ts,
                    flight_time=T,
                    speed_scale=float(args.speed_scale),
                    grid_size=int(args.search_grid_size),
                    refine_steps=int(args.search_refine_steps),
                    refine_width=float(args.search_refine_width),
                )
                speed = speed_choice["speed"]
                print(f"  Policy target xy: {np.round(used_target_xy, 4)}")
                print(
                    f"  Speed strategy: requested={args.speed_strategy}, used={speed_choice['strategy_used']}, "
                    f"speed={speed:.4f} m/s"
                )
                print(
                    f"  Policy speed={speed_choice['policy_speed']:.4f} m/s "
                    f"(pred err {speed_choice['policy_error_pred_m']:.4f} m)"
                )
                if speed_choice["search_best_speed"] is not None:
                    print(
                        f"  Search best speed={speed_choice['search_best_speed']:.4f} m/s "
                        f"(pred err {speed_choice['search_error_pred_m']:.4f} m)"
                    )

                throw_result = execute_throw(
                    client_id=client,
                    arm=arm,
                    policy_target_xy=used_target_xy,
                    gt_target_xy=gt_target_xy,
                    release_pos=release_pos,
                    speed=speed,
                    dt=Ts,
                    flight_time=T,
                    hit_tolerance=args.hit_tolerance,
                    target_radius=args.target_radius,
                    direct_mode=args.direct,
                    slow_factor=args.slow,
                    draw_debug=args.draw_debug,
                    camera_cfg=camera_cfg,
                    frame_callback=(video_recorder.write_bgr if video_recorder is not None else None),
                )

                if throw_result["landing"] is not None:
                    landing = throw_result["landing"]
                    print(
                        f"  Landing xyz: {np.round(landing, 4)} "
                        f"| gt err={throw_result['landing_error_gt_m']:.4f} m "
                        f"| est err={throw_result['landing_error_est_m']:.4f} m"
                    )
                else:
                    print("  Landing: unavailable")

            row = {
                "throw_idx": throw_idx + 1,
                "gt_target_x": float(gt_target_world[0]),
                "gt_target_y": float(gt_target_world[1]),
                "gt_target_z": float(gt_target_world[2]),
                "detection_found": chosen_detection is not None,
                "detection_confidence": None if chosen_detection is None else float(chosen_detection["confidence"]),
                "detection_class_name": None if chosen_detection is None else chosen_detection["class_name"],
                "bbox_x1": None if chosen_detection is None else float(chosen_detection["bbox_xyxy"][0]),
                "bbox_y1": None if chosen_detection is None else float(chosen_detection["bbox_xyxy"][1]),
                "bbox_x2": None if chosen_detection is None else float(chosen_detection["bbox_xyxy"][2]),
                "bbox_y2": None if chosen_detection is None else float(chosen_detection["bbox_xyxy"][3]),
                "depth_surface_m": None if depth_info is None else float(depth_info["surface_depth_m"]),
                "est_target_x": None if est_target_world is None else float(est_target_world[0]),
                "est_target_y": None if est_target_world is None else float(est_target_world[1]),
                "est_target_z": None if est_target_world is None else float(est_target_world[2]),
                "perception_error_m": (
                    None if est_target_world is None else float(np.linalg.norm(est_target_world[:2] - gt_target_xy))
                ),
                "surface_vs_vp_error_m": None if depth_info is None else float(depth_info["surface_vs_vp_error_m"]),
                "used_target_x": None if used_target_xy is None else float(used_target_xy[0]),
                "used_target_y": None if used_target_xy is None else float(used_target_xy[1]),
                "used_ground_truth_fallback": used_gt_fallback,
                "speed_strategy_requested": args.speed_strategy,
                "speed_strategy_used": None if throw_result is None else speed_choice["strategy_used"],
                "policy_speed_mps": None if throw_result is None else speed_choice["policy_speed"],
                "policy_predicted_error_m": None if throw_result is None else speed_choice["policy_error_pred_m"],
                "search_best_speed_mps": None if throw_result is None else speed_choice["search_best_speed"],
                "search_predicted_error_m": None if throw_result is None else speed_choice["search_error_pred_m"],
                "planned_vx": None if throw_result is None else float(throw_result["v_cmd"][0]),
                "planned_vy": None if throw_result is None else float(throw_result["v_cmd"][1]),
                "planned_vz": None if throw_result is None else float(throw_result["v_cmd"][2]),
                "achieved_speed_mps": (
                    None
                    if throw_result is None
                    else float(np.linalg.norm(throw_result["v_achieved"]))
                ),
                "release_time_s": None if throw_result is None else float(throw_result["release_time"]),
                "landing_x": None if throw_result is None or throw_result["landing"] is None else float(throw_result["landing"][0]),
                "landing_y": None if throw_result is None or throw_result["landing"] is None else float(throw_result["landing"][1]),
                "landing_z": None if throw_result is None or throw_result["landing"] is None else float(throw_result["landing"][2]),
                "landing_error_gt_m": None if throw_result is None else throw_result["landing_error_gt_m"],
                "landing_error_est_m": None if throw_result is None else throw_result["landing_error_est_m"],
                "hit_gt": False if throw_result is None else bool(throw_result["hit_gt"]),
                "hit_est": False if throw_result is None else bool(throw_result["hit_est"]),
                "skipped_throw": skipped_throw,
                "skip_reason": skip_reason,
            }
            rows.append(row)

            if throw_result is not None and args.draw_debug:
                time.sleep(1.0 * args.slow if not args.direct else 0.0)
                if video_recorder is not None:
                    tail_frame = make_camera_frame_bgr(client, camera_cfg, args.direct)
                    for _ in range(max(1, int(round(args.record_fps * 0.5)))):
                        video_recorder.write_bgr(tail_frame)

            p.removeBody(target_id, physicsClientId=client)
            if throw_result is not None:
                p.removeBody(throw_result["ball_id"], physicsClientId=client)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(rows)

        detected = sum(int(bool(row["detection_found"])) for row in rows)
        thrown = sum(int(not row["skipped_throw"]) for row in rows)
        hits = sum(int(bool(row["hit_gt"])) for row in rows)
        perception_errors = [row["perception_error_m"] for row in rows if row["perception_error_m"] is not None]
        print("\nRun complete.")
        print(f"  Detections: {detected}/{len(rows)}")
        print(f"  Throws executed: {thrown}/{len(rows)}")
        print(f"  Ground-truth hits: {hits}/{thrown if thrown > 0 else 1}")
        if perception_errors:
            print(f"  Mean perception xy error: {np.mean(perception_errors):.4f} m")
        print(f"  CSV log: {csv_path}")

        if args.keep_window_open and not args.direct:
            try:
                while p.isConnected(client):
                    p.stepSimulation(physicsClientId=client)
                    time.sleep(1.0 / 240.0)
            except KeyboardInterrupt:
                pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if video_recorder is not None:
            video_recorder.close()
        if log_id is not None:
            p.stopStateLogging(log_id, physicsClientId=client)
        if p.isConnected(client):
            p.disconnect(client)
        if old_sigint is not None:
            try:
                signal.signal(signal.SIGINT, old_sigint)
            except Exception:
                pass


if __name__ == "__main__":
    main()
