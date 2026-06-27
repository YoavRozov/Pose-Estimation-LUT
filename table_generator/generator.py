import warp as wp

from apriltag import AprilTag, get_corner
from camera import Camera, corner_in_frame, project_point

@wp.func
def robot_pose_to_camera_pose(x: wp.float32, y: wp.float32, theta: wp.float32,
                                cam_to_robot: wp.transform) -> wp.transform:
    """Builds the camera's world/field pose from a 2D robot pose (x, y, heading)
    and the fixed camera-to-robot extrinsic transform."""
    robot_pos = wp.vec3(x, y, 0.0)
    robot_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), theta)
    robot_pose_world = wp.transform(robot_pos, robot_rot)
    return wp.transform_multiply(robot_pose_world, cam_to_robot)

@wp.kernel
def sweep_tag_corners(
    xs: wp.array(dtype=wp.float32), # type: ignore
    ys: wp.array(dtype=wp.float32), # type: ignore
    thetas: wp.array(dtype=wp.float32), # type: ignore
    tag: AprilTag,
    cam_to_robot: wp.transform,
    fx: wp.float32, fy: wp.float32, cx: wp.float32, cy: wp.float32,
    img_w: wp.float32, img_h: wp.float32, max_range: wp.float32,
    out_corners: wp.array2d(dtype=wp.float32),   # (N, 8) # type: ignore
    out_valid: wp.array(dtype=wp.int32),       # (N,) # type: ignore
):
    """
    Projects the corners of a single AprilTag into the camera's pixel frame for a sweep of robot poses.
    
    Args:
        xs: wp.array  x-coordinates of robot poses (meters)
        ys: wp.array  y-coordinates of robot poses (meters)
        thetas: wp.array  headings of robot poses (radians)
        tag: AprilTag  the AprilTag to project
        cam_to_robot: wp.transform  the fixed camera-to-robot extrinsic transform
        fx, fy, cx, cy: wp.float32  camera intrinsic parameters
        img_w, img_h: wp.float32  image dimensions
        max_range: wp.float32  maximum range for range checks
        out_corners: wp.array2d  output array of projected corner pixel coordinates (N, 8)
        out_valid: wp.array  output array of validity flags (N,)
    """
    tid = wp.tid()

    cam_pose_world = robot_pose_to_camera_pose(xs[tid], ys[tid], thetas[tid], cam_to_robot)

    cam = Camera()
    cam.fx = fx
    cam.fy = fy
    cam.cx = cx
    cam.cy = cy
    cam.world_to_cam = wp.transform_inverse(cam_pose_world)

    valid = wp.int32(1)

    for i in range(4):
        corner_world = get_corner(tag, i)
        pixel, z_ok = project_point(cam, corner_world)

        if z_ok == 0:
            valid = 0

        # range check uses the camera-space depth, so recompute it here
        corner_cam = wp.transform_point(cam.world_to_cam, corner_world)
        if corner_cam[2] > max_range:
            valid = 0

        if corner_in_frame(pixel, img_w, img_h) == 0:
            valid = 0

        out_corners[tid, 2 * i] = pixel[0]
        out_corners[tid, 2 * i + 1] = pixel[1]

    out_valid[tid] = valid