"""Generate high-resolution multi-panel Zoom-In Error visualization.

Replicates and enhances the layout of docs/zoom_block.png, specifically focusing
on the held-out "left" station under Spatial Leave-One-Station-Out (SLOSO) validation.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Ensure backend modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from backend.dataset import load_dataset
from backend.features import build_features
from backend.models import make_model
from backend.stations import get_stations_for_bbox, match_stations_to_grid


def main(target_station_query: str = "kensington") -> Path:
    base_dir = Path(__file__).resolve().parents[2]
    docs_dir = base_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset (London benchmark)
    key = "london_2026-09-20_2026-09-26_s0.050_c60"
    data = load_dataset(key)
    feat = build_features(data)

    lats = np.asarray(data["lats"], dtype=float)
    lons = np.asarray(data["lons"], dtype=float)
    clats = np.asarray(data["clats"], dtype=float)
    clons = np.asarray(data["clons"], dtype=float)
    block_idx = np.asarray(data["block_idx"])

    # 2. Fit Spatial LightGBM model
    model = make_model("lightgbm")
    train_sel = feat["y_valid"].ravel()
    X_flat = feat["X"].reshape(-1, feat["X"].shape[-1])[train_sel]
    y_flat = feat["y"].ravel()[train_sel]
    model.fit(X_flat, y_flat)

    # Predict full field
    pred_ratio = np.clip(model.predict(feat["X"].reshape(-1, feat["X"].shape[-1])), -1.6, 1.6)
    c_up = feat["c_up"]
    pred = (c_up.ravel() * np.exp(pred_ratio)).reshape(c_up.shape)
    ref = np.asarray(data["ref"], dtype=float)

    # Time-mean fields across the week
    pred_mean = np.nanmean(pred, axis=0)
    ref_mean = np.nanmean(ref, axis=0)
    coarse_mean = np.nanmean(c_up, axis=0)
    err_mean = pred_mean - ref_mean

    # 3. Locate target left station
    stations = get_stations_for_bbox(data["bbox"])
    matched = match_stations_to_grid(stations, lats, lons)

    target_match = None
    for m in matched:
        if target_station_query.lower() in m["station"].name.lower() or target_station_query.lower() in m["station"].station_id.lower():
            target_match = m
            break
    if target_match is None:
        # Default to westernmost / left station
        matched_sorted = sorted(matched, key=lambda x: x["station"].longitude)
        target_match = matched_sorted[0]

    st = target_match["station"]
    st_row, st_col = target_match["row"], target_match["col"]
    st_block = int(block_idx[st_row, st_col])

    # Find the bounding sub-grid for this coarse block
    r_idx, c_idx = np.where(block_idx == st_block)
    r_min, r_max = int(r_idx.min()), int(r_idx.max()) + 1
    c_min, c_max = int(c_idx.min()), int(c_idx.max()) + 1

    sub_lats = lats[r_min:r_max]
    sub_lons = lons[c_min:c_max]
    sub_pred = pred_mean[r_min:r_max, c_min:c_max]
    sub_ref = ref_mean[r_min:r_max, c_min:c_max]
    sub_err = err_mean[r_min:r_max, c_min:c_max]
    sub_coarse_val = float(np.nanmean(coarse_mean[r_min:r_max, c_min:c_max]))

    sub_rmse = float(np.sqrt(np.mean(sub_err**2)))
    sub_mae = float(np.mean(np.abs(sub_err)))

    # Coarse grid center for this block
    b_lat = float(np.mean(sub_lats))
    b_lon = float(np.mean(sub_lons))

    # 4. Create the Multi-Panel Figure matching zoom_block.png layout
    fig = plt.figure(figsize=(18, 9), facecolor="white")

    # Title Banner
    fig.suptitle(
        f"AERO-SHARP: Spatial Leave-One-Station-Out (SLOSO) Zoom-In Accuracy | Left Station: {st.name} ({st.station_id})\n"
        f"Coarse satellite pixel: {sub_coarse_val:.1f} µg/m³ uniform | Inside ML resolves {sub_pred.min():.1f}–{sub_pred.max():.1f} µg/m³ | "
        f"Ground station obs: {sub_ref.mean():.1f} µg/m³ avg (Zoom MAE: {sub_mae:.2f} µg/m³, RMSE: {sub_rmse:.2f} µg/m³)",
        fontsize=12,
        fontweight="bold",
        y=0.97,
    )

    # Top Plot: Region Overview with red zoom box and station locations
    ax_top = fig.add_axes([0.55, 0.52, 0.38, 0.38])
    im_top = ax_top.imshow(
        coarse_mean,
        origin="lower",
        extent=[lons.min(), lons.max(), lats.min(), lats.max()],
        cmap="viridis",
        aspect="auto",
    )
    ax_top.set_title(
        f"Week-mean coarse satellite field (0.25°) — Red box = Left Station ({st.name} @ {st.latitude:.3f}°N, {st.longitude:.3f}°E)",
        fontsize=10,
        pad=8,
    )
    ax_top.set_xlabel("longitude (°E)", fontsize=9)
    ax_top.set_ylabel("latitude (°N)", fontsize=9)

    # Plot all physical ground stations
    for m in matched:
        s = m["station"]
        is_target = s.station_id == st.station_id
        ax_top.scatter(
            s.longitude,
            s.latitude,
            c="#ff3333" if is_target else "#00e5ff",
            edgecolors="black",
            s=80 if is_target else 40,
            marker="*" if is_target else "o",
            zorder=6,
        )
        if is_target:
            ax_top.annotate(
                f" ★ {s.name}",
                (s.longitude, s.latitude),
                color="#ffff00",
                fontweight="bold",
                fontsize=8,
                textcoords="offset points",
                xytext=(8, -3),
                bbox=dict(boxstyle="round,pad=0.2", fc="#222222", ec="#ffff00", alpha=0.85),
            )

    # Red box around the zoomed coarse block
    box_lon_min, box_lon_max = float(sub_lons.min() - 0.025), float(sub_lons.max() + 0.025)
    box_lat_min, box_lat_max = float(sub_lats.min() - 0.025), float(sub_lats.max() + 0.025)
    rect = patches.Rectangle(
        (box_lon_min, box_lat_min),
        box_lon_max - box_lon_min,
        box_lat_max - box_lat_min,
        linewidth=2.5,
        edgecolor="#ff0000",
        facecolor="none",
        zorder=5,
    )
    ax_top.add_patch(rect)

    cb_top = fig.colorbar(im_top, ax=ax_top, fraction=0.035, pad=0.03)
    cb_top.set_label("µg/m³", fontsize=9)

    # Bottom Row of 4 Subplots
    # Shared color bounds for concentrations
    vmin_c = np.floor(min(sub_pred.min(), sub_ref.min(), sub_coarse_val) - 0.5)
    vmax_c = np.ceil(max(sub_pred.max(), sub_ref.max(), sub_coarse_val) + 0.5)

    sub_extent = [box_lon_min, box_lon_max, box_lat_min, box_lat_max]

    # Panel 1: Coarse satellite block (uniform blurred view)
    ax1 = fig.add_axes([0.05, 0.08, 0.18, 0.35])
    coarse_block_arr = np.full_like(sub_pred, sub_coarse_val)
    im1 = ax1.imshow(coarse_block_arr, origin="lower", extent=sub_extent, cmap="viridis", vmin=vmin_c, vmax=vmax_c, aspect="auto")
    ax1.set_title(f"One coarse pixel (0.25°)\n(blurred satellite view): {sub_coarse_val:.1f} µg/m³", fontsize=10, pad=6)
    ax1.text(
        (box_lon_min + box_lon_max) / 2,
        (box_lat_min + box_lat_max) / 2,
        f"{sub_coarse_val:.1f}",
        color="white",
        fontsize=24,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.3, edgecolor="none"),
    )
    ax1.set_xlabel("longitude (°E)", fontsize=9)
    ax1.set_ylabel("latitude (°N)", fontsize=9)
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_label("µg/m³", fontsize=8)

    # Panel 2: ML Zoom-In (Spatial LightGBM, 5x5 sub-cells)
    ax2 = fig.add_axes([0.28, 0.08, 0.18, 0.35])
    im2 = ax2.imshow(sub_pred, origin="lower", extent=sub_extent, cmap="viridis", vmin=vmin_c, vmax=vmax_c, aspect="auto")
    ax2.set_title("ML zoom-in (0.05 deg, 5×5 sub-cells)\nSpatial LightGBM (Gradient Boosting)", fontsize=10, pad=6)
    ax2.set_xlabel("longitude (°E)", fontsize=9)

    # Annotate values on 5x5 grid
    h_sub, w_sub = sub_pred.shape
    d_lon = (box_lon_max - box_lon_min) / w_sub
    d_lat = (box_lat_max - box_lat_min) / h_sub
    for r in range(h_sub):
        for c in range(w_sub):
            x = box_lon_min + (c + 0.5) * d_lon
            y = box_lat_min + (r + 0.5) * d_lat
            val = sub_pred[r, c]
            ax2.text(x, y, f"{val:.1f}", color="white" if val < (vmin_c + vmax_c) / 2 else "black", fontsize=8, ha="center", va="center")
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label("µg/m³", fontsize=8)

    # Panel 3: Independent Reference & Physical Ground Station
    ax3 = fig.add_axes([0.51, 0.08, 0.18, 0.35])
    im3 = ax3.imshow(sub_ref, origin="lower", extent=sub_extent, cmap="viridis", vmin=vmin_c, vmax=vmax_c, aspect="auto")
    ax3.set_title(f"Independent ground reference (0.1°)\n★ {st.name}", fontsize=10, pad=6)
    ax3.set_xlabel("longitude (°E)", fontsize=9)

    # Annotate reference values
    for r in range(h_sub):
        for c in range(w_sub):
            x = box_lon_min + (c + 0.5) * d_lon
            y = box_lat_min + (r + 0.5) * d_lat
            val = sub_ref[r, c]
            ax3.text(x, y, f"{val:.1f}", color="white" if val < (vmin_c + vmax_c) / 2 else "black", fontsize=8, ha="center", va="center")

    # Star at exact station coordinate
    ax3.scatter(st.longitude, st.latitude, c="#ff2222", edgecolors="white", s=130, marker="*", zorder=10)
    ax3.annotate(f" {st.station_id}", (st.longitude, st.latitude), color="white", fontsize=8, fontweight="bold")
    cb3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cb3.set_label("µg/m³", fontsize=8)

    # Panel 4: Error map (ML - reference)
    ax4 = fig.add_axes([0.74, 0.08, 0.18, 0.35])
    max_err = max(0.5, float(np.nanmax(np.abs(sub_err))))
    max_err_rounded = np.ceil(max_err * 2.0) / 2.0
    im4 = ax4.imshow(sub_err, origin="lower", extent=sub_extent, cmap="coolwarm", vmin=-max_err_rounded, vmax=max_err_rounded, aspect="auto")
    ax4.set_title(f"Error: ML – reference (SLOSO)\nMAE: {sub_mae:.2f} µg/m³ · RMSE: {sub_rmse:.2f} µg/m³", fontsize=10, pad=6)
    ax4.set_xlabel("longitude (°E)", fontsize=9)

    # Annotate error values with plus/minus sign
    for r in range(h_sub):
        for c in range(w_sub):
            x = box_lon_min + (c + 0.5) * d_lon
            y = box_lat_min + (r + 0.5) * d_lat
            val = sub_err[r, c]
            sign = "+" if val > 0 else ""
            txt_color = "black" if abs(val) < max_err_rounded * 0.65 else "white"
            ax4.text(x, y, f"{sign}{val:.2f}", color=txt_color, fontsize=8, ha="center", va="center", fontweight="bold")

    ax4.scatter(st.longitude, st.latitude, c="#ff2222", edgecolors="black", s=100, marker="*", zorder=10)
    cb4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cb4.set_label("error (µg/m³)", fontsize=8)

    # Save to docs/zoom_station_<slug>_error.png and docs/zoom_block.png
    slug = st.name.lower().replace(" ", "_").replace(".", "")
    out_file1 = docs_dir / f"zoom_station_{slug}_error.png"
    out_file2 = docs_dir / "zoom_station_left_error.png"
    out_file_main = docs_dir / "zoom_block.png"
    fig.savefig(out_file1, dpi=140, bbox_inches="tight")
    fig.savefig(out_file2, dpi=140, bbox_inches="tight")
    fig.savefig(out_file_main, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"Successfully generated zoom-in error graphic:")
    print(f"  Target Station: {st.name} ({st.station_id})")
    print(f"  Zoomed-in Coarse Block: lat={b_lat:.3f}, lon={b_lon:.3f}")
    print(f"  Sub-pixel RMSE: {sub_rmse:.3f} µg/m³")
    print(f"  Sub-pixel MAE:  {sub_mae:.3f} µg/m³")
    print(f"  Saved to: {out_file_main}")
    print(f"  Saved to: {out_file1}")
    return out_file_main


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "kensington"
    main(query)
