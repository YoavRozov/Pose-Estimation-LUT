import yaml
import numpy as np
import h5py
import warp as wp

from field_layout import load_field_layout
from camera import Camera, project_point
from apriltag import AprilTag, get_corner
from generator import sweep_tag_corners

from helper import build_cam_to_robot, compute_sweep_box, host_mirror, pick_sample_pose_near_tag, run_single_pose_check
from visualize import plot_field_layout, plot_sweep_regions, plot_projection_crosscheck

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_pose_grid(bounds: dict, grid: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full rectangular grid over (x, y, theta). See note below on optimizing this."""
    xs_1d = np.arange(bounds["x_min"], bounds["x_max"], grid["xy_resolution"])
    ys_1d = np.arange(bounds["y_min"], bounds["y_max"], grid["xy_resolution"])
    thetas_1d = np.arange(0.0, 360.0, grid["theta_resolution_deg"]) * np.pi / 180.0

    xx, yy, tt = np.meshgrid(xs_1d, ys_1d, thetas_1d, indexing="ij")
    return xx.ravel().astype(np.float32), yy.ravel().astype(np.float32), tt.ravel().astype(np.float32)


def main(config_path: str):
    wp.init()
    cfg = load_config(config_path)
    cam_cfg = cfg["camera"]
    tags = load_field_layout(cfg["field_layout_json"], cam_cfg["tag_size"])
    cam_to_robot = build_cam_to_robot(cfg["cam_to_robot"])

    # --- VISUAL CHECKPOINT 1: field layout ---
    plot_field_layout(host_mirror(tags), cfg["field_bounds"])

    # --- VISUAL CHECKPOINT 2: sweep regions ---
    sweep_boxes = {tid: compute_sweep_box(t, cfg["camera"]["max_range"], cfg["field_bounds"])
                   for tid, t in tags.items()}
    plot_sweep_regions(host_mirror(tags), cfg["field_bounds"], sweep_boxes)

    # --- VISUAL CHECKPOINT 3: projection cross-check on one sample tag/pose ---
    sample_tag_id = next(iter(tags))
    sample_pose = pick_sample_pose_near_tag(tags[sample_tag_id], cfg["field_bounds"])
    sample_corners = run_single_pose_check(tags[sample_tag_id], cam_to_robot, cam_cfg, sample_pose)
    plot_projection_crosscheck(host_mirror(tags)[sample_tag_id], cam_to_robot, cam_cfg,
                                 sample_pose, sample_corners)

    confirm = input("\nDo all 3 visualizations look correct? Type 'yes' to proceed "
                     "with full dataset generation: ").strip().lower()
    if confirm != "yes":
        print("Aborted. Fix the flagged issue and re-run.")
        return

    xs_np, ys_np, thetas_np = build_pose_grid(cfg["field_bounds"], cfg["grid"])
    n = len(xs_np)
    print(f"Grid size: {n:,} poses per tag")

    xs = wp.array(xs_np, dtype=wp.float32)
    ys = wp.array(ys_np, dtype=wp.float32)
    thetas = wp.array(thetas_np, dtype=wp.float32)

    for tag_id, tag in tags.items():
        out_corners = wp.zeros((n, 8), dtype=wp.float32)
        out_valid = wp.zeros(n, dtype=wp.int32)

        wp.launch(
            kernel=sweep_tag_corners,
            dim=n,
            inputs=[
                xs, ys, thetas, tag, cam_to_robot,
                cam_cfg["fx"], cam_cfg["fy"], cam_cfg["cx"], cam_cfg["cy"],
                float(cam_cfg["image_width"]), float(cam_cfg["image_height"]),
                cam_cfg["max_range"],
            ],
            outputs=[out_corners, out_valid],
        )

        valid_mask = out_valid.numpy().astype(bool)
        n_valid = valid_mask.sum()
        print(f"Tag {tag_id}: {n_valid:,} / {n:,} valid poses")

        corners_np = out_corners.numpy()[valid_mask]
        poses_np = np.stack([xs_np, ys_np, thetas_np], axis=1)[valid_mask]

        out_path = f"{cfg['output_dir']}/tag_{tag_id}.h5"
        with h5py.File(out_path, "w") as f:
            f.create_dataset("poses", data=poses_np, compression="gzip")
            f.create_dataset("corners", data=corners_np, compression="gzip")
        print(f"  -> wrote {out_path}")


if __name__ == "__main__":
    # import sys
    # main(sys.argv[1])
    main("table_generator/config/configuration.yaml")