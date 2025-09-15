# Codes: Wainwright Risk Index and Autoencoder

This folder contains notebooks to build a feature-enriched grid for Wainwright, compute a rule-based risk index, and train a small convolutional autoencoder to reconstruct/predict the risk index from other layers. It also produces figures and a GeoPackage with predictions and errors.

## Directory contents

- `helper_00.ipynb` — Build the enriched 50 m grid by joining inputs (infrastructure, erosion) and sampling raster InSAR (excess ice). Saves `Wainwright/grids/grid50_with_all_data.shp`.
- `calculate_risk.ipynb` — Compute a rule-based `risk_index` using domain rules and neighbor effects; writes `Wainwright/grid_50_risk_index.shp`.
- `autoencoder.ipynb` — Autoencoder experiments:
  - Regression of continuous `risk_index` from 6 feature channels.
  - Alternative: one-hot risk categories as targets.
- `autoencoder_02.ipynb` — Streamlined, CPU-safe autoencoder that trains on the full scene with a masked loss, evaluates, and writes outputs (images + GeoPackage). Recommended.
- `helper_00.html` — HTML export (large) of a helper notebook run.
- Output artifacts produced here:
  - `wainwright_risk_original.png`
  - `wainwright_risk_reconstructed.png`
  - `wainwright_risk_abs_error.png`
  - `wainwright_risk_original_pred_error.png`
  - `wainwright_risk.gpkg` (GeoPackage with predictions/errors)

## Data dependencies (paths as used in notebooks)

- Grid base and outputs
  - `Wainwright/grids/grids_50m.shp` (input base grid)
  - `Wainwright/grids/grid50_with_all_data.shp` (enriched grid)
  - `Wainwright/grid_50_risk_index.shp` (rule-based risk output)
- Vector layers
  - `Wainwright/infrastructure/detailed_infrastructure/w_infrastructure/w_detailed_infrastructure.shp`
  - `Wainwright/erosion/W_erosion_forecast/W_erosion_forecast.shp`
- Raster layer
  - `data/sar/Wainwright/e_mean_period.tif` (InSAR; sampled into `excess_ice`)

Notes:
- The notebooks reproject inputs to `EPSG:3338` if needed.
- Shapefile field-name truncation may occur; downstream code expects columns like `infra_exis`, `erosion_ex`, `ice_wedge_`.

## Environment setup

Tested with Python 3.10. Create and activate a virtual environment, then install packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install geopandas shapely pyogrio rasterio rasterstats numpy pandas matplotlib seaborn plotly scikit-learn scipy tensorflow networkx
```

If system libraries are needed for Geo stack, consider installing via Conda or using manylinux wheels. On some systems, `rasterio`/`geopandas` require GDAL/PROJ installed.

### Run TensorFlow on CPU (avoid CUDA issues)

TensorFlow GPU errors can be bypassed by hiding GPUs. Both autoencoder notebooks include a CPU-only setup; if not, set before importing/using TF:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # or "-1"
```

## Workflow

1) Build enriched grid (optional if already present)
- Open `helper_00.ipynb` and run all cells. It:
  - Loads base grid `Wainwright/grids/grids_50m.shp` and reprojects to `EPSG:3338`.
  - Joins infrastructure and erosion layers (spatial intersects).
  - Samples InSAR raster to compute `excess_ice` via zonal statistics.
  - Saves `Wainwright/grids/grid50_with_all_data.shp`.

2) Compute rule-based risk index
- Open `calculate_risk.ipynb` and run all cells. It:
  - Loads `Wainwright/grids/grid50_with_all_data_clipped.shp` or similar enriched grid.
  - Applies rules: ice wedges (+0.25), excess ice ≥ 0.2 (+0.25), drained lakes in [0, 0.5] (+0.25), infrastructure present or neighbor (×0.5), neighbor to contaminated “Open” (×1.5), and sets no-go (1.0) if erosion or contaminated “Open”.
  - Clips final `risk_index` to [0, 1].
  - Saves `Wainwright/grid_50_risk_index.shp`.

3) Train and evaluate autoencoder (recommended: `autoencoder_02.ipynb`)
- Open `autoencoder_02.ipynb` and run all cells. It:
  - Rasterizes the irregular grid to a 2D image stack at 50 m resolution with channels: `drained_la`, `excess_ice`, `ice_wedge_`, `contaminat_open` (binary), `infra_exis`, `erosion_ex`.
  - Uses a small encoder–decoder CNN to regress continuous `risk_index` with a masked MSE loss (ignores missing targets).
  - Evaluates on a random pixel hold-out, writes maps and scatter diagnostics, and saves predictions/errors back to a GeoPackage `wainwright_risk.gpkg`.

Alternative: `autoencoder.ipynb` also includes a categorical (one-hot) risk approach in addition to continuous regression.

## Key columns expected

- Predictors: `drained_la`, `excess_ice`, `ice_wedge_` (0/1), `contaminat` (mapped to `contaminat_open`), `infra_exis` (0/1), `erosion_ex` (0/1)
- Target: `risk_index` (continuous in [0, 1])

## Outputs produced here

- Figures: `wainwright_risk_original.png`, `wainwright_risk_reconstructed.png`, `wainwright_risk_abs_error.png`, `wainwright_risk_original_pred_error.png`
- Vector data: `wainwright_risk.gpkg` with `risk_pred` and `risk_abs_err`

## Tips and troubleshooting

- Memory: The current notebooks train on a single full-scene batch. If you hit RAM issues, reduce model width (filters), use fewer epochs, or tile the scene.
- CRS: Inputs are auto-reprojected to `EPSG:3338`. Ensure all custom inputs are reprojectable.
- Shapefile field truncation: When saving Shapefiles, long names may be truncated (e.g., `infra_exist` → `infra_exis`). The autoencoder notebooks use the truncated names.
- Reproducibility: Random splits are seeded in `autoencoder_02.ipynb` for evaluation; training determinism is not enforced by default.

## Citation/acknowledgments

If you use these notebooks in a publication, please cite the data sources (InSAR, infrastructure, erosion datasets) and this repository. Update this section with your preferred citation.
