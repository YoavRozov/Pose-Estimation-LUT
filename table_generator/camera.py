from dataclasses import dataclass

import warp as wp
import numpy as np

@wp.struct
class Camera:
    """Camera intrinsics + extrinsics (world -> camera space).
    
    The camera's pose is defined in the world/field frame, and the camera's coordinate system is defined as:
        x: right
        y: down
        z: forward
    """
    fx: wp.float32
    fy: wp.float32
    cx: wp.float32
    cy: wp.float32
    world_to_cam: wp.transform   # inverse of camera's pose in world/field frame

def make_camera(fx: wp.float32, fy: wp.float32, cx: wp.float32, cy: wp.float32, cam_pose_world: wp.transform) -> Camera:
    """
    Creates a Camera object with the given intrinsics and pose in world space.

    Args:
        fx: wp.float32  focal length in x (pixels)
        fy: wp.float32  focal length in y (pixels)
        cx: wp.float32  principal point x (pixels)
        cy: wp.float32  principal point y (pixels)
        cam_pose_world: wp.transform  camera's pose in world/field frame
    """
    cam = Camera()
    cam.fx = fx
    cam.fy = fy
    cam.cx = cx
    cam.cy = cy
    cam.world_to_cam = wp.transform_inverse(cam_pose_world)
    return cam

@wp.func
def project_point(cam: Camera, point_world: wp.vec3):
    """
    Projects a 3D world-space point into the camera's pixel frame.

    Args:
        cam: Camera  camera object
        point_world: wp.vec3  3D point in world/field space

    Returns:
        pixel: wp.vec2   (px, py) — only meaningful if valid == 1
        valid: wp.int32  1 if point is in front of the camera (z > 0), else 0
    """
    # transform point from world/field space into camera space
    point_cam = wp.transform_point(cam.world_to_cam, point_world)

    valid = wp.int32(1)

    # check if point is in front of the camera (z > 0)
    if point_cam[2] <= 0.0:
        valid = 0

    # avoid div-by-zero if behind/at the camera plane
    z = wp.max(point_cam[2], 1.0e-6)

    px = cam.fx * (point_cam[0] / z) + cam.cx
    py = cam.fy * (point_cam[1] / z) + cam.cy

    pixel = wp.vec2(px, py)
    return pixel, valid

@wp.func
def corner_in_frame(pixel: wp.vec2, img_w: wp.float32, img_h: wp.float32) -> wp.int32:
    if pixel[0] < 0.0 or pixel[0] > img_w or pixel[1] < 0.0 or pixel[1] > img_h:
        return 0
    return 1