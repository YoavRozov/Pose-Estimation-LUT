import numpy as np
import matplotlib.pyplot as plt
import cv2

from helper import compute_extrinsics_cv2

def plot_field_layout(tags: dict, field_bounds: dict):
    """Top-down view of all tags with their facing direction.
    Expects tags from host_mirror(): each entry has 'center' (3,) and 'facing' (2,)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for tag_id, tag in tags.items():
        c = tag["center"]
        facing = tag["facing"]   # already a unit (fx, fy) vector, no quaternion needed

        ax.plot(c[0], c[1], 'o', color='tab:blue', markersize=8)
        ax.annotate(str(tag_id), (c[0], c[1]), textcoords="offset points",
                    xytext=(6, 6), fontsize=9)

        ax.arrow(c[0], c[1], facing[0] * 0.5, facing[1] * 0.5,
                  head_width=0.15, color='tab:red')

    ax.set_xlim(field_bounds["x_min"], field_bounds["x_max"])
    ax.set_ylim(field_bounds["y_min"], field_bounds["y_max"])
    ax.set_aspect("equal")
    ax.set_title("Field layout: tag positions + facing direction (red arrow)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    plt.show()


def plot_sweep_regions(tags: dict, field_bounds: dict, sweep_boxes: dict):
    """sweep_boxes: {tag_id: (x_min, x_max, y_min, y_max)} - the per-tag region
    actually being swept (from the 7m range-bound optimization)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for tag_id, tag in tags.items():
        c = tag["center"]
        ax.plot(c[0], c[1], 'o', color='tab:blue', markersize=8)
        ax.annotate(str(tag_id), (c[0], c[1]), textcoords="offset points", xytext=(6, 6))

        x_min, x_max, y_min, y_max = sweep_boxes[tag_id]
        ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     fill=False, edgecolor='tab:orange', linewidth=1))

    ax.set_xlim(field_bounds["x_min"], field_bounds["x_max"])
    ax.set_ylim(field_bounds["y_min"], field_bounds["y_max"])
    ax.set_aspect("equal")
    ax.set_title("Per-tag sweep regions (orange boxes) — check these cover\n"
                 "expected scoring areas without being wastefully oversized")
    plt.show()


def plot_projection_crosscheck(tag: dict, cam_to_robot, cam_cfg: dict,
                                 sample_pose: tuple, warp_corners: np.ndarray):
    """
    warp_corners: (8,) array from your Warp sweep kernel for sample_pose, this tag.
    Recomputes the same projection independently via cv2.projectPoints and overlays both.
    """
    x, y, theta = sample_pose

    # build robot->world->camera extrinsics the cv2 way, independently from the Warp path
    robot_rot_z = cv2.Rodrigues(np.array([0, 0, theta]))[0]
    # ... compose with cam_to_robot translation/rotation (host-side numpy, mirroring
    # your wp.transform_multiply logic) to get rvec/tvec for cv2.projectPoints
    rvec, tvec = compute_extrinsics_cv2(x, y, theta, cam_to_robot)

    camera_matrix = np.array([
        [cam_cfg["fx"], 0, cam_cfg["cx"]],
        [0, cam_cfg["fy"], cam_cfg["cy"]],
        [0, 0, 1],
    ])

    corners_world = np.array([tag["corners"][0], tag["corners"][1], tag["corners"][2], tag["corners"][3]])
    cv2_corners, _ = cv2.projectPoints(corners_world, rvec, tvec, camera_matrix, None)
    cv2_corners = cv2_corners.reshape(4, 2)

    warp_corners = warp_corners.reshape(4, 2)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.invert_yaxis()  # image coordinates: y grows downward
    ax.add_patch(plt.Rectangle((0, 0), cam_cfg["image_width"], cam_cfg["image_height"],
                                 fill=False, edgecolor='black'))

    ax.plot(*warp_corners.T, 'o-', color='tab:blue', label='Warp projection', markersize=10)
    ax.plot(*cv2_corners.T, 'x--', color='tab:red', label='cv2.projectPoints', markersize=10)
    for i in range(4):
        ax.annotate(str(i), warp_corners[i], color='tab:blue')

    ax.legend()
    ax.set_title(f"Tag {tag['id']} corner projection cross-check at pose "
                 f"x={x:.2f} y={y:.2f} θ={np.degrees(theta):.1f}°\n"
                 "Blue and red should overlap — mismatch = convention bug")
    plt.show()


def quat_rotate_2d(q, v):
    """Rotates a 3D vector by quaternion q=(x,y,z,w), returns full 3D result."""
    x, y, z, w = q
    qv = np.array([x, y, z])
    t = 2 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)