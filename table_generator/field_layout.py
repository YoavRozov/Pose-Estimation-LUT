import json
import warp as wp
from apriltag import AprilTag, make_apriltag  # from your existing module


def load_field_layout(json_path: str, tag_size: float) -> dict[int, AprilTag]:
    with open(json_path, "r") as f:
        data = json.load(f)

    tags = {}
    for entry in data["tags"]:
        tag_id = entry["ID"]
        t = entry["pose"]["translation"]
        q = entry["pose"]["rotation"]["quaternion"]

        center = wp.vec3(t["x"], t["y"], t["z"])
        # WPILib JSON is scalar-first (W,X,Y,Z) -> Warp quat is (x,y,z,w)
        rotation = wp.quat(q["X"], q["Y"], q["Z"], q["W"])

        tags[tag_id] = make_apriltag(tag_id, center, rotation, tag_size)

    return tags