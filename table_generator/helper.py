import math
import numpy as np
import cv2
import warp as wp

from apriltag import AprilTag


# ---------- independent numpy quaternion math (deliberately NOT reusing wp functions) ----------

def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """q = (x, y, z, w). Standard quaternion -> rotation matrix, implemented from scratch
    so this cross-check path doesn't share code with the Warp side."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def compute_extrinsics_cv2(x: float, y: float, theta: float, cam_to_robot: wp.transform):
    """
    Independently recomputes the world->camera extrinsics for cv2.projectPoints,
    mirroring robot_pose_to_camera_pose's math but via plain numpy, as a true
    second implementation to catch convention bugs in the Warp path.

    Returns: rvec, tvec suitable for cv2.projectPoints
    """
    # wp.transform is indexable as [px, py, pz, qx, qy, qz, qw]
    c2r_t = np.array([cam_to_robot[0], cam_to_robot[1], cam_to_robot[2]])
    c2r_q = np.array([cam_to_robot[3], cam_to_robot[4], cam_to_robot[5], cam_to_robot[6]])
    R_c2r = quat_to_rotmat(c2r_q)

    # robot pose in world frame: position (x, y, 0), rotation theta about +Z
    R_robot = quat_to_rotmat(np.array([0.0, 0.0, math.sin(theta / 2.0), math.cos(theta / 2.0)]))
    t_robot = np.array([x, y, 0.0])

    # cam_pose_world = robot_pose_world * cam_to_robot
    R_cam_world = R_robot @ R_c2r
    t_cam_world = R_robot @ c2r_t + t_robot

    # cv2.projectPoints wants world->camera (the inverse)
    R_world_cam = R_cam_world.T
    t_world_cam = -R_world_cam @ t_cam_world

    rvec, _ = cv2.Rodrigues(R_world_cam)
    tvec = t_world_cam.reshape(3, 1)
    return rvec, tvec


# ---------- the five requested functions ----------

import numpy as np

# Fixed body->optical axis remap. NOT measured — this encodes the camera
# convention itself: body forward -> optical Z, body left -> optical -X,
# body up -> optical -Y.
_BODY_TO_OPTICAL_MATRIX = np.array([
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
    [1.0,  0.0,  0.0],
])

def matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Standard rotation-matrix -> quaternion (x,y,z,w) conversion (Shepperd's method),
    used here rather than a hand-derived constant to avoid arithmetic mistakes."""
    m = R
    trace = m[0,0] + m[1,1] + m[2,2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2,1] - m[1,2]) * s
        y = (m[0,2] - m[2,0]) * s
        z = (m[1,0] - m[0,1]) * s
    elif m[0,0] > m[1,1] and m[0,0] > m[2,2]:
        s = 2.0 * math.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2])
        w = (m[2,1] - m[1,2]) / s
        x = 0.25 * s
        y = (m[0,1] + m[1,0]) / s
        z = (m[0,2] + m[2,0]) / s
    elif m[1,1] > m[2,2]:
        s = 2.0 * math.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2])
        w = (m[0,2] - m[2,0]) / s
        x = (m[0,1] + m[1,0]) / s
        y = 0.25 * s
        z = (m[1,2] + m[2,1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1])
        w = (m[1,0] - m[0,1]) / s
        x = (m[0,2] + m[2,0]) / s
        y = (m[1,2] + m[2,1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


_OPTICAL_FIX_QUAT = matrix_to_quat_xyzw(_BODY_TO_OPTICAL_MATRIX.T)

def build_cam_to_robot(c2r_cfg: dict) -> wp.transform:
    """
    Builds the camera-to-robot extrinsic, composing:
      1. the measured physical mounting transform (position + any tilt/pan,
         expressed in body axes: X-forward, Y-left, Z-up)
      2. the fixed body<->optical axis remap (NOT measured, a convention constant)
    """
    mounting = wp.transform(
        wp.vec3(*c2r_cfg["translation"]),
        wp.quat(*c2r_cfg["rotation_quaternion"]),
    )
    optical_fix = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(*_OPTICAL_FIX_QUAT))
    return wp.transform_multiply(mounting, optical_fix)


def host_mirror(tags: dict) -> dict:
    """
    Pulls plain numpy data back out of AprilTag Warp structs for plotting.

    Note: AprilTag only stores the 4 world-space corners, not a separate
    center/quaternion — so 'center' and 'facing' here are *derived* from the
    corners (center = mean of corners, facing = the tag face normal projected
    into the XY plane), rather than stored fields. This means plot_field_layout
    should use tag["facing"] directly instead of calling quat_rotate_2d.
    """
    mirrored = {}
    for tag_id, tag in tags.items():
        corners = np.array([list(getattr(tag, f"corner{i}")) for i in range(4)])
        center = corners.mean(axis=0)

        edge1 = corners[1] - corners[0]
        edge2 = corners[3] - corners[0]
        normal = np.cross(edge1, edge2)

        facing_xy = normal[:2]
        norm = np.linalg.norm(facing_xy)
        facing_xy = facing_xy / norm if norm > 1e-9 else np.array([1.0, 0.0])

        mirrored[tag_id] = {
            "id": tag_id,
            "center": center,
            "corners": corners,
            "facing": facing_xy,
        }
    return mirrored


def compute_sweep_box(tag: "AprilTag", max_range: float, field_bounds: dict) -> tuple:
    """(x_min, x_max, y_min, y_max) region around this tag worth sweeping,
    clipped to the field, based on the range cutoff."""
    corners = np.array([list(getattr(tag, f"corner{i}")) for i in range(4)])
    center = corners.mean(axis=0)

    x_min = max(field_bounds["x_min"], center[0] - max_range)
    x_max = min(field_bounds["x_max"], center[0] + max_range)
    y_min = max(field_bounds["y_min"], center[1] - max_range)
    y_max = min(field_bounds["y_max"], center[1] + max_range)
    return (x_min, x_max, y_min, y_max)


def pick_sample_pose_near_tag(tag: "AprilTag", field_bounds: dict, distance: float = 3.0) -> tuple:
    """
    Picks a robot pose ~`distance` meters in front of the tag's face, heading
    so the tag is roughly centered in view — a reasonable sample pose for the
    projection cross-check (NOT guaranteed in-frame for your exact cam_to_robot
    offset; that's exactly what the cross-check plot will reveal).
    """
    corners = np.array([list(getattr(tag, f"corner{i}")) for i in range(4)])
    center = corners.mean(axis=0)

    edge1 = corners[1] - corners[0]
    edge2 = corners[3] - corners[0]
    normal = np.cross(edge1, edge2)
    facing_xy = normal[:2]
    norm = np.linalg.norm(facing_xy)
    facing_xy = facing_xy / norm if norm > 1e-9 else np.array([1.0, 0.0])

    sample_xy = center[:2] + facing_xy * distance
    sample_xy[0] = np.clip(sample_xy[0], field_bounds["x_min"] + 0.1, field_bounds["x_max"] - 0.1)
    sample_xy[1] = np.clip(sample_xy[1], field_bounds["y_min"] + 0.1, field_bounds["y_max"] - 0.1)

    dx = center[0] - sample_xy[0]
    dy = center[1] - sample_xy[1]
    theta = math.atan2(dy, dx)  # heading pointed back toward the tag

    return (float(sample_xy[0]), float(sample_xy[1]), theta)


def run_single_pose_check(tag: "AprilTag", cam_to_robot: wp.transform, cam_cfg: dict,
                            sample_pose: tuple) -> np.ndarray:
    """Launches the real sweep_tag_corners kernel for a single pose, so the
    cross-check validates the actual production kernel, not a re-implementation."""
    from generator import sweep_tag_corners  # avoid circular import at module load

    x, y, theta = sample_pose
    xs = wp.array([x], dtype=wp.float32)
    ys = wp.array([y], dtype=wp.float32)
    thetas = wp.array([theta], dtype=wp.float32)
    out_corners = wp.zeros((1, 8), dtype=wp.float32)
    out_valid = wp.zeros(1, dtype=wp.int32)

    wp.launch(
        kernel=sweep_tag_corners,
        dim=1,
        inputs=[
            xs, ys, thetas, tag, cam_to_robot,
            cam_cfg["fx"], cam_cfg["fy"], cam_cfg["cx"], cam_cfg["cy"],
            float(cam_cfg["image_width"]), float(cam_cfg["image_height"]),
            cam_cfg["max_range"],
        ],
        outputs=[out_corners, out_valid],
    )

    if out_valid.numpy()[0] == 0:
        print("Warning: sample pose was flagged invalid (corner out of frame or "
              "behind camera) — the cross-check plot may show a degenerate projection. "
              "Consider adjusting `distance` in pick_sample_pose_near_tag.")

    return out_corners.numpy()[0]