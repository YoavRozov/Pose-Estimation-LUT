import warp as wp

@wp.struct
class AprilTag:
    """An AprilTag's ID and the world-space positions of its 4 corners."""
    tag_id: wp.int32
    corner0: wp.vec3   # convention: top-left
    corner1: wp.vec3   # top-right
    corner2: wp.vec3   # bottom-right
    corner3: wp.vec3   # bottom-left


@wp.func
def get_corner(tag: AprilTag, i: wp.int32):
    """Indexed accessor so kernels can loop `for i in range(4): get_corner(tag, i)`."""
    if i == 0:
        return tag.corner0
    elif i == 1:
        return tag.corner1
    elif i == 2:
        return tag.corner2
    else:
        return tag.corner3

def make_apriltag(tag_id: int, center: wp.vec3, rotation: wp.quat, tag_size: float) -> AprilTag:
    """
    center: tag's center position in field/world frame
    rotation: tag's orientation in field/world frame (face normal direction)
    tag_size: full edge length of the tag in meters (e.g. 0.1651 for 6.5in tags)
    """
    half = tag_size / 2.0

    # TODO: verify that the corners are defined in the same order as the AprilTag library's output.
    # Tag is mounted vertically: local face lies in the XZ plane, with local
    # +Y as the outward-facing normal. Z varies (top vs bottom edge of tag);
    # X varies (left vs right), which then gets rotated by yaw (theta about Z).
    local_corners = [
        wp.vec3(0.0,  half,  half),  # top-left
        wp.vec3(0.0, -half,  half),  # top-right
        wp.vec3(0.0, -half, -half),  # bottom-right
        wp.vec3(0.0,  half, -half),  # bottom-left
    ]

    tag = AprilTag()
    tag.tag_id = tag_id
    world_corners = [wp.quat_rotate(rotation, c) + center for c in local_corners]
    tag.corner0, tag.corner1, tag.corner2, tag.corner3 = world_corners
    return tag

@wp.func
def get_tag_normal(tag: AprilTag) -> wp.vec3:
    """
    World-space outward-facing normal, derived from the corner winding —
    consistent with the local +X = outward convention fixed earlier in
    make_apriltag (cross(corner1-corner0, corner3-corner0) reproduces that
    same +X direction after rotation, since cross products transform
    correctly under proper rotations).
    """
    edge1 = tag.corner1 - tag.corner0
    edge2 = tag.corner3 - tag.corner0
    normal = wp.cross(edge1, edge2)
    return wp.normalize(normal)