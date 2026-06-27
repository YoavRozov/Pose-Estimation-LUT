import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.spatial import cKDTree

from field_layout import load_field_layout
from main import load_config


def load_dataset(tag_id: int, data_dir: str):
    path = f"{data_dir}/tag_{tag_id}.h5"
    with h5py.File(path, "r") as f:
        poses = f["poses"][:]      # (N, 3) -> x, y, theta
        corners = f["corners"][:]  # (N, 8)
    return poses, corners


# ---------------- COVERAGE (static) ----------------

def compute_coverage_grid(poses: np.ndarray, field_bounds: dict, xy_resolution: float):
    """
    Bins all valid rows into (x,y) cells and counts how many valid theta
    samples landed in each cell. A cell with a suspiciously low count near
    a tag's range (but not at the edge of max_range) usually indicates a
    bug, not real geometry -- real coverage should shrink gradually with
    distance/angle, not have sharp holes.
    """
    x_edges = np.arange(field_bounds["x_min"], field_bounds["x_max"] + xy_resolution, xy_resolution)
    y_edges = np.arange(field_bounds["y_min"], field_bounds["y_max"] + xy_resolution, xy_resolution)

    counts, _, _ = np.histogram2d(poses[:, 0], poses[:, 1], bins=[x_edges, y_edges])
    return counts, x_edges, y_edges


def plot_coverage_heatmap(counts: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray,
                            tag_center: tuple, tag_id: int):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(counts.T, origin="lower",
                    extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                    aspect="equal", cmap="viridis")
    fig.colorbar(im, ax=ax, label="valid θ samples per (x,y) cell")
    ax.plot(tag_center[0], tag_center[1], 'r*', markersize=15, label="tag")
    ax.legend()
    ax.set_title(f"Tag {tag_id}: coverage density\n"
                 "Look for sharp holes/rings — real falloff with range/angle should be smooth")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    plt.show()


def find_gap_cells(counts: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray,
                     tag_center: tuple, max_range: float, min_expected_ratio: float = 0.1):
    """
    Flags (x,y) cells that are within max_range of the tag (so SHOULD have
    some valid heading coverage) but have suspiciously few or zero valid
    samples, after excluding cells right at the range boundary (where low
    counts are expected and correct).
    """
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    max_count = counts.max()

    suspicious = []
    for i, x in enumerate(x_centers):
        for j, y in enumerate(y_centers):
            dist = np.hypot(x - tag_center[0], y - tag_center[1])
            # only flag cells comfortably inside range, not near the boundary
            if dist < max_range * 0.85:
                if counts[i, j] < max_count * min_expected_ratio:
                    suspicious.append((round(x, 2), round(y, 2), int(counts[i, j]), round(dist, 2)))

    if suspicious:
        print(f"\n{len(suspicious)} suspicious low/zero-coverage cells found "
              f"(well within range, but low count):")
        for x, y, c, d in suspicious[:30]:
            print(f"  x={x}, y={y}  count={c}  dist_to_tag={d}m")
        if len(suspicious) > 30:
            print(f"  ... and {len(suspicious) - 30} more")
    else:
        print("No suspicious gap cells found within 85% of max range.")
    return suspicious


# ---------------- INTERACTIVE (per-pose inspection) ----------------

def launch_interactive_viewer(poses: np.ndarray, corners: np.ndarray, tag_corners_world: np.ndarray,
                                tag_center: tuple, field_bounds: dict, cam_cfg: dict, tag_id: int):
    """
    Sliders for x, y, theta snap to the nearest ACTUAL grid values present in
    the dataset (not arbitrary continuous values) so you're always querying
    real generated data, not interpolating across a gap.
    """
    xs_unique = np.unique(poses[:, 0])
    ys_unique = np.unique(poses[:, 1])
    thetas_unique = np.unique(poses[:, 2])

    tree = cKDTree(poses)

    fig, (ax_img, ax_map) = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(bottom=0.25)

    def query_and_draw(x, y, theta):
        dist, idx = tree.query([x, y, theta])
        matched_pose = poses[idx]
        matched_corners = corners[idx].reshape(4, 2)

        # --- left: simulated camera frame ---
        ax_img.clear()
        ax_img.set_xlim(0, cam_cfg["image_width"])
        ax_img.set_ylim(cam_cfg["image_height"], 0)  # image y grows downward
        ax_img.add_patch(plt.Rectangle((0, 0), cam_cfg["image_width"], cam_cfg["image_height"],
                                         fill=False, edgecolor="black"))
        loop = list(range(4)) + [0]
        ax_img.plot(matched_corners[loop, 0], matched_corners[loop, 1], 'o-', color="tab:blue")
        for i in range(4):
            ax_img.annotate(str(i), matched_corners[i], color="tab:blue")
        ax_img.set_title(f"Projected corners\nmatched pose: x={matched_pose[0]:.2f} "
                          f"y={matched_pose[1]:.2f} θ={np.degrees(matched_pose[2]):.1f}°\n"
                          f"query distance: {dist:.3f} {'(EXACT)' if dist < 1e-4 else '(NEAREST -- possible gap)'}")
        ax_img.set_aspect("equal")

        # --- right: top-down field map ---
        ax_map.clear()
        ax_map.set_xlim(field_bounds["x_min"], field_bounds["x_max"])
        ax_map.set_ylim(field_bounds["y_min"], field_bounds["y_max"])
        ax_map.set_aspect("equal")
        ax_map.plot(tag_center[0], tag_center[1], 'r*', markersize=15, label="tag")
        ax_map.plot(matched_pose[0], matched_pose[1], 'o', color="tab:blue", markersize=8)
        heading_dx = 0.4 * np.cos(matched_pose[2])
        heading_dy = 0.4 * np.sin(matched_pose[2])
        ax_map.arrow(matched_pose[0], matched_pose[1], heading_dx, heading_dy,
                      head_width=0.15, color="tab:blue")
        ax_map.set_title(f"Tag {tag_id}: robot pose (top-down)")
        ax_map.legend()

        fig.canvas.draw_idle()

    ax_x = plt.axes([0.15, 0.13, 0.7, 0.03])
    ax_y = plt.axes([0.15, 0.08, 0.7, 0.03])
    ax_theta = plt.axes([0.15, 0.03, 0.7, 0.03])

    s_x = Slider(ax_x, "x (m)", xs_unique.min(), xs_unique.max(), valinit=xs_unique[len(xs_unique)//2])
    s_y = Slider(ax_y, "y (m)", ys_unique.min(), ys_unique.max(), valinit=ys_unique[len(ys_unique)//2])
    s_theta = Slider(ax_theta, "θ (deg)", 0, 359,
                       valinit=np.degrees(thetas_unique[len(thetas_unique)//2]))

    def on_change(_):
        query_and_draw(s_x.val, s_y.val, np.radians(s_theta.val))

    s_x.on_changed(on_change)
    s_y.on_changed(on_change)
    s_theta.on_changed(on_change)

    query_and_draw(s_x.val, s_y.val, np.radians(s_theta.val))
    plt.show()


# ---------------- CLI ----------------
def get_tag_center(tags: dict, tag_id: int) -> tuple:
    """Pulls the real tag center from the parsed field layout (mean of its
    4 known corners), not approximated from dataset contents."""
    tag = tags[tag_id]
    corners = np.array([list(getattr(tag, f"corner{i}")) for i in range(4)])
    center = corners.mean(axis=0)
    return (float(center[0]), float(center[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=int, required=True)
    parser.add_argument("--mode", choices=["coverage", "interactive"], required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    field_bounds = cfg["field_bounds"]
    cam_cfg = cfg["camera"]
    xy_resolution = cfg["grid"]["xy_resolution"]
    max_range = cam_cfg["max_range"]
    data_dir = cfg["output_dir"]

    tags = load_field_layout(cfg["field_layout_json"], cam_cfg["tag_size"])
    if args.tag not in tags:
        raise ValueError(f"Tag {args.tag} not found in field layout "
                          f"({cfg['field_layout_json']}). Available IDs: {sorted(tags.keys())}")

    tag_center = get_tag_center(tags, args.tag)

    poses, corners = load_dataset(args.tag, data_dir)

    if args.mode == "coverage":
        counts, x_edges, y_edges = compute_coverage_grid(poses, field_bounds, xy_resolution)
        plot_coverage_heatmap(counts, x_edges, y_edges, tag_center, args.tag)
        find_gap_cells(counts, x_edges, y_edges, tag_center, max_range)
    else:
        launch_interactive_viewer(poses, corners, None, tag_center, field_bounds, cam_cfg, args.tag)


if __name__ == "__main__":
    main()