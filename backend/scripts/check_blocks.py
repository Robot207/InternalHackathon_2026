import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from backend.grids import block_upsample, make_grid  # noqa: E402

art = Path(__file__).resolve().parents[1] / "artifacts" / "latest"
layers = json.loads((art / "layers.json").read_text())

meta = json.loads((art / "meta.json").read_text())
from backend.dataset import dataset_paths  # noqa: E402

_, sp = dataset_paths(meta["summary_key"])
summary = json.loads(sp.read_text())

grid = make_grid(summary["bbox"], float(summary["fine_step"]))
coarse = np.array(layers["layers"]["coarse"][12], dtype=float)  # frame 12
bilin = np.array(layers["layers"]["coarse_bilinear"][12], dtype=float)
pred = np.array(layers["layers"]["prediction"][12], dtype=float)
gap = np.array(layers["layers"]["cloud_gap"][12], dtype=float)

# block constancy check
flat_idx = grid["block_idx"].ravel()
ok = True
for b in range(grid["n_blocks"]):
    vals = coarse.ravel()[flat_idx == b]
    if np.nanstd(vals) > 1e-9:
        ok = False
print("coarse block-constant:", ok)
print("coarse vs bilinear max diff:", round(float(np.nanmax(np.abs(coarse - bilin))), 3))
print("shapes:", coarse.shape, pred.shape, "range coarse:", layers["ranges"]["coarse"],
      "range coarse_bilinear:", layers["ranges"]["coarse_bilinear"])
print("gap block values (unique):", sorted(set(np.round(gap.ravel(), 3).tolist()))[:8])

# render previews
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, (arr, title) in zip(
    axes,
    [(coarse, "coarse 0.25 BLOCKS (new left)"), (bilin, "coarse bilinear (old left)"), (pred, "prediction (right)")],
):
    im = ax.imshow(arr, origin="lower", cmap="viridis", aspect="auto")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
out = Path(__file__).resolve().parents[2] / "docs" / "block_check.png"
fig.tight_layout()
fig.savefig(out, dpi=110)
print("saved", out)
