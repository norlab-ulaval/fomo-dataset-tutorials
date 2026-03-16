"""
fomo_get_to_know.py
===================
Standalone Python script version of the FoMo "Get to Know the Dataset" tutorial.

This script is the companion to the Jupyter notebook fomo_get_to_know.ipynb.
It can be run directly without Jupyter:

    python fomo_get_to_know.py

Requirements:
    pip install boto3 pandas numpy matplotlib Pillow

The FoMo dataset is publicly accessible -- no AWS credentials required.

Citation:
    Boxan et al. (2026). FoMo: A Multi-Season Dataset for Robot Navigation
    in Foret Montmorency. arXiv:2603.08433 [cs.RO].
    https://arxiv.org/abs/2603.08433
"""

# ── Built-in libraries ─────────────────────────────────────────────────────
import io
import json
from pprint import pprint

# ── Third-party libraries ──────────────────────────────────────────────────
import boto3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from botocore import UNSIGNED
from botocore.config import Config
from PIL import Image

# ── Compatibility shim: display() works in Jupyter but not in plain Python ─
try:
    display
except NameError:
    def display(x):
        print(x)

# ── S3 client (unsigned = public access, no AWS account required) ──────────
BUCKET = "fomo-dataset"
DATA_PREFIX = "data/"
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

print("Setup complete. Connected to S3 bucket:", BUCKET)
print("Data prefix:", DATA_PREFIX)


# =============================================================================
# Helper functions
# =============================================================================

def list_prefixes(prefix="", delimiter="/"):
    """List all common prefixes (i.e., 'folders') under a given prefix."""
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, Delimiter=delimiter)
    return [p["Prefix"] for p in resp.get("CommonPrefixes", [])]


def get_csv(key):
    """Download and parse a CSV file from S3 into a pandas DataFrame."""
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def load_lidar_scan(key):
    """
    Load a FoMo lidar point cloud from S3.
    Returns an (N, 6) float32 array: [x, y, z, intensity, ring_id, timestamp]
    Each point is 6 float32 values = 24 bytes.
    """
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    data = obj["Body"].read()
    # Truncate to nearest multiple of 24 bytes (6 float32 values per point)
    point_size = 4 * 6  # 4 bytes per float32, 6 fields per point
    remainder = len(data) % point_size
    if remainder != 0:
        data = data[:len(data) - remainder]
    raw = np.frombuffer(data, dtype=np.float32)
    return raw.reshape(-1, 6)


def load_radar_scan(key):
    """Load a Navtech radar polar image from S3 as a numpy array."""
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    img = Image.open(io.BytesIO(obj["Body"].read()))
    return np.array(img)


def load_ground_truth(session_prefix):
    """Load the TUM-format ground truth trajectory."""
    obj = s3.get_object(Bucket=BUCKET, Key=f"{session_prefix}gt.txt")
    content = obj["Body"].read().decode("utf-8")
    rows = []
    for line in content.strip().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        rows.append([float(v) for v in line.split()])
    return pd.DataFrame(rows, columns=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])


def load_calib(session_prefix, filename):
    obj = s3.get_object(Bucket=BUCKET, Key=f"{session_prefix}calib/{filename}")
    return json.loads(obj["Body"].read().decode("utf-8"))


# =============================================================================
# Listing deployments and sessions
# =============================================================================

deployments = list_prefixes(prefix=DATA_PREFIX)
print(f"\nFound {len(deployments)} deployments:\n")
for d in deployments:
    print(" ", d)

# ── Dynamically find the green session in the Jan 29, 2025 deployment ──────
DEPLOYMENT = "data/2025-01-29/"
sessions = list_prefixes(prefix=DEPLOYMENT)

print(f"\nSessions in deployment '{DEPLOYMENT}':\n")
for s in sessions:
    print(" ", s)

SESSION = next((s for s in sessions if "green" in s), sessions[0])
print(f"\nUsing session: {SESSION}")


# =============================================================================
# Q3: How do I download and load data?
# =============================================================================

# ── 3a. Loading CSV metadata ──────────────────────────────────────────────

print("\n--- 3a. Meteo & Snow data ---")
meteo_df = get_csv(f"{SESSION}metadata/meteo_data.csv")
snow_df  = get_csv(f"{SESSION}metadata/snow_data.csv")

print("Meteo data columns:", list(meteo_df.columns))
print(f"\nMeteo data -- {len(meteo_df)} rows")
print(meteo_df.head())

print(f"\nSnow data -- {len(snow_df)} rows")
print(snow_df.head())


# ── 3b. Loading IMU data ──────────────────────────────────────────────────

print("\n--- 3b. IMU data ---")
imu_df = get_csv(f"{SESSION}vectornav.csv")

print("IMU columns:", list(imu_df.columns))
print(f"Total IMU samples: {len(imu_df)}")
print(imu_df.head())


# ── 3c. Loading lidar point clouds (.bin) ─────────────────────────────────

print("\n--- 3c. Lidar point cloud ---")
lidar_prefix = f"{SESSION}robosense/"
resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=lidar_prefix, MaxKeys=3)
first_scan_key = resp["Contents"][0]["Key"]

print("Loading scan:", first_scan_key)
points = load_lidar_scan(first_scan_key)
print(f"Point cloud shape: {points.shape}  ->  {points.shape[0]:,} points")
print("\nFirst 5 points (x, y, z, intensity, ring_id, timestamp):")
print(points[:5])


# ── 3d. Loading radar scans (.png) ────────────────────────────────────────

print("\n--- 3d. Radar scan ---")
radar_prefix = f"{SESSION}navtech/"
resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=radar_prefix, MaxKeys=3)
first_radar_key = resp["Contents"][0]["Key"]

radar_img = load_radar_scan(first_radar_key)
print(f"Radar image shape: {radar_img.shape}  ->  {radar_img.shape[0]} azimuths x {radar_img.shape[1]} columns")
print(f"Range bins (R): {radar_img.shape[1] - 11}")


# ── 3e. Loading the Ground Truth trajectory ───────────────────────────────

print("\n--- 3e. Ground truth trajectory ---")
gt_df = load_ground_truth(SESSION)
print(f"Ground truth: {len(gt_df)} poses")
print(gt_df.head())

dx = np.diff(gt_df["x"].values)
dy = np.diff(gt_df["y"].values)
total_dist = np.sum(np.sqrt(dx**2 + dy**2))
print(f"\nApproximate trajectory length: {total_dist:.1f} m")


# ── 3f. Loading calibration files ─────────────────────────────────────────

print("\n--- 3f. Calibration files ---")
transforms = load_calib(SESSION, "transforms.json")
print("Available transforms (sensor frame pairs):")
if isinstance(transforms, dict):
    pprint(list(transforms.keys()))
elif isinstance(transforms, list):
    pprint(transforms)
else:
    pprint(transforms)


# =============================================================================
# Q4: Visuals
# =============================================================================

# ── 4a. Ground truth trajectory -- top-down view ──────────────────────────

print("\n--- 4a. Plotting ground truth trajectory ---")
fig, ax = plt.subplots(figsize=(7, 7), dpi=100, facecolor="white")

x = gt_df["x"] - gt_df["x"].iloc[0]
y = gt_df["y"] - gt_df["y"].iloc[0]

t_norm = (gt_df["t"] - gt_df["t"].iloc[0]) / (gt_df["t"].iloc[-1] - gt_df["t"].iloc[0])
sc = ax.scatter(x, y, c=t_norm, cmap="viridis", s=2, zorder=2)
ax.plot(x.iloc[0], y.iloc[0], "go", markersize=10, label="Start", zorder=3)
ax.plot(x.iloc[-1], y.iloc[-1], "rs", markersize=10, label="End", zorder=3)

cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
cbar.set_label("Time progression", fontsize=10)

ax.set_xlabel("X (m)", fontsize=11)
ax.set_ylabel("Y (m)", fontsize=11)
ax.set_title("Ground Truth Trajectory\nGreen - 2025-01-29 (Winter, -19 degC)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_aspect("equal")
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.3)
ax.set_facecolor("#f8f9fa")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("4a_ground_truth_trajectory.png", dpi=100)
plt.show()
print("Saved: 4a_ground_truth_trajectory.png")


# ── 4b. Lidar point cloud -- top-down bird's-eye view ─────────────────────

print("\n--- 4b. Plotting lidar BEV ---")
mask = (
    (np.abs(points[:, 0]) < 40) &
    (np.abs(points[:, 1]) < 40) &
    (points[:, 2] > -2.0) &
    (points[:, 2] < 5.0)
)
pts = points[mask]

fig, ax = plt.subplots(figsize=(8, 8), dpi=100, facecolor="#0d0d0d")
sc = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2],
                cmap="plasma", s=0.3, alpha=0.8, vmin=-1.5, vmax=3.0)

cbar = plt.colorbar(sc, ax=ax, shrink=0.5)
cbar.set_label("Height Z (m)", color="white", fontsize=10)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.get_yticklabels(), color="white")

ax.set_xlabel("X (m)", color="white", fontsize=11)
ax.set_ylabel("Y (m)", color="white", fontsize=11)
ax.set_title("RoboSense Ruby Plus - Single Scan (BEV)\nGreen - 2025-01-29",
             color="white", fontsize=13, fontweight="bold", pad=12)
ax.set_aspect("equal")
ax.set_facecolor("#0d0d0d")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")

plt.tight_layout()
plt.savefig("4b_lidar_bev.png", dpi=100)
plt.show()
print(f"Plotted {pts.shape[0]:,} points (out of {points.shape[0]:,} total)")
print("Saved: 4b_lidar_bev.png")


# ── 4c. Radar scan -- polar image ─────────────────────────────────────────

print("\n--- 4c. Plotting radar polar image ---")
range_data = radar_img[:, 11:].astype(np.float32)

fig, ax = plt.subplots(figsize=(10, 4), dpi=100, facecolor="white")
im = ax.imshow(range_data, aspect="auto", cmap="inferno",
               interpolation="nearest", vmin=0, vmax=255)
plt.colorbar(im, ax=ax, shrink=0.8, label="Intensity")

ax.set_xlabel("Range bin", fontsize=11)
ax.set_ylabel("Azimuth index", fontsize=11)
ax.set_title("Navtech CIR-304H - Radar Polar Image\nGreen - 2025-01-29",
             fontsize=13, fontweight="bold", pad=12)

plt.tight_layout()
plt.savefig("4c_radar_polar.png", dpi=100)
plt.show()
print(f"Polar image: {range_data.shape[0]} azimuths x {range_data.shape[1]} range bins")
print("Saved: 4c_radar_polar.png")


# ── 4d. Seasonal temperature & snow depth overview ────────────────────────

print("\n--- 4d. Plotting seasonal conditions ---")
deployment_info = [
    {"date": "Nov 21",  "temp": -3,  "snow_cm": 24,  "conditions": "Clouds"},
    {"date": "Nov 28",  "temp": -5,  "snow_cm": 40,  "conditions": "Clear"},
    {"date": "Jan 10",  "temp": -15, "snow_cm": 60,  "conditions": "Clear"},
    {"date": "Jan 29",  "temp": -19, "snow_cm": 72,  "conditions": "Clear"},
    {"date": "Mar 10",  "temp": -8,  "snow_cm": 120, "conditions": "Snowfall"},
    {"date": "Apr 15",  "temp": 4,   "snow_cm": 30,  "conditions": "Rain"},
    {"date": "May 28",  "temp": 12,  "snow_cm": 0,   "conditions": "Clear"},
    {"date": "Jun 26",  "temp": 18,  "snow_cm": 0,   "conditions": "Clear"},
    {"date": "Aug 20",  "temp": 18,  "snow_cm": 0,   "conditions": "Clear"},
    {"date": "Sep 24",  "temp": 10,  "snow_cm": 0,   "conditions": "Clear"},
    {"date": "Oct 14",  "temp": 5,   "snow_cm": 0,   "conditions": "Clear"},
    {"date": "Nov 03",  "temp": 2,   "snow_cm": 0,   "conditions": "Night/Rain"},
]
df_meta = pd.DataFrame(deployment_info)
dates = df_meta["date"]
x = np.arange(len(dates))

fig, ax1 = plt.subplots(figsize=(13, 5), dpi=100, facecolor="white")
ax2 = ax1.twinx()

colors = ["#3b82f6" if t < 0 else "#f97316" for t in df_meta["temp"]]
ax1.bar(x - 0.2, df_meta["temp"], width=0.35, color=colors, alpha=0.85,
        label="Avg. Temperature (degC)", zorder=2)

ax2.plot(x, df_meta["snow_cm"], "o--", color="#60a5fa", linewidth=2,
         markersize=7, label="Snow Depth (cm)", zorder=3)
ax2.fill_between(x, df_meta["snow_cm"], alpha=0.12, color="#60a5fa")

ax1.axhline(0, color="black", linewidth=0.7, linestyle="-")
ax1.set_xticks(x)
ax1.set_xticklabels(dates, rotation=30, ha="right", fontsize=10)
ax1.set_ylabel("Temperature (degC)", fontsize=11)
ax2.set_ylabel("Snow Depth (cm)", fontsize=11, color="#3b82f6")
ax2.tick_params(axis="y", colors="#3b82f6")
ax1.set_title("FoMo Deployment Conditions - Temperature & Snow Depth",
              fontsize=13, fontweight="bold", pad=12)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

ax1.set_facecolor("#f8f9fa")
ax1.grid(True, linestyle="--", alpha=0.3, axis="y")
for spine in ["top"]:
    ax1.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("4d_seasonal_conditions.png", dpi=100)
plt.show()
print("Saved: 4d_seasonal_conditions.png")


# ── 4e. Power consumption across seasons ──────────────────────────────────

print("\n--- 4e. Plotting motor current draw ---")
current_left_df  = get_csv(f"{SESSION}metadata/current_left.csv")
current_right_df = get_csv(f"{SESSION}metadata/current_right.csv")

t_col = current_left_df.columns[0]
c_col = current_left_df.columns[1]

t_left  = current_left_df[t_col].values
t_right = current_right_df[t_col].values
t0 = min(t_left[0], t_right[0])
t_left_s  = (t_left  - t0) / 1e6
t_right_s = (t_right - t0) / 1e6

fig, ax = plt.subplots(figsize=(12, 4), dpi=100, facecolor="white")
ax.plot(t_left_s,  current_left_df[c_col],  label="Left motor",  alpha=0.8, linewidth=0.8)
ax.plot(t_right_s, current_right_df[c_col], label="Right motor", alpha=0.8, linewidth=0.8)

ax.set_xlabel("Time (s)", fontsize=11)
ax.set_ylabel("Current (A)", fontsize=11)
ax.set_title("Motor Current Draw - Green Trajectory, Jan 29 2025 (-19 degC, Tracks)",
             fontsize=12, fontweight="bold", pad=10)
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.3)
ax.set_facecolor("#f8f9fa")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("4e_motor_current.png", dpi=100)
plt.show()

print(f"Peak current - Left:  {current_left_df[c_col].max():.1f} A")
print(f"Peak current - Right: {current_right_df[c_col].max():.1f} A")
print("Saved: 4e_motor_current.png")


# =============================================================================
# Q5: Localization performance across deployments
# =============================================================================

print("\n--- Q5. Localization performance ---")
deployments_eval = ["Nov21", "Nov28", "Jan29", "Mar10", "Jun26", "Aug20", "Sep24", "Oct14", "Nov03"]

results = {
    "Proprioceptive":       [3.7,  4.6,  15.3, 4.6,  3.9, 3.4,  5.0,  3.7,  4.6],
    "Lidar-Inertial":       [4.5,  4.8,  3.8,  4.3,  11.7, 4.0, 25.4, 46.5, 4.3],
    "Radar-Gyro T&R":       [8.1, 14.2,  16.8, 32.3,  8.8, 4.2,  9.3, 24.6, 9.3],
    "Stereo-Inertial SLAM": [1.2,  1.5,  2.4,  7.6,   8.8, 1.3,  2.3,  0.9, None],
}

df_results = pd.DataFrame(results, index=deployments_eval)

fig, ax = plt.subplots(figsize=(13, 5), dpi=100, facecolor="white")

x = np.arange(len(deployments_eval))
width = 0.2
colors_methods = ["#6366f1", "#f59e0b", "#10b981", "#ef4444"]

for i, (method, color) in enumerate(zip(df_results.columns, colors_methods)):
    vals = [v if v is not None else 0 for v in df_results[method]]
    ax.bar(x + i * width - 1.5 * width, vals, width,
           label=method, color=color, alpha=0.85, zorder=2)
    for j, v in enumerate(df_results[method]):
        if v is None:
            ax.text(x[j] + i * width - 1.5 * width, 2, "N/A",
                    ha="center", va="bottom", fontsize=7, color="gray", rotation=90)

ax.axhline(5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="5% drift threshold")
ax.set_xticks(x)
ax.set_xticklabels(deployments_eval, fontsize=10)
ax.set_ylabel("Mean Translation Drift (%)", fontsize=11)
ax.set_title("Localization Performance Across Deployments\n(Map built from Jan 29, 2025)",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=9, loc="upper left")
ax.set_facecolor("#f8f9fa")
ax.grid(True, linestyle="--", alpha=0.3, axis="y")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("q5_localization_performance.png", dpi=100)
plt.show()
print("Saved: q5_localization_performance.png")

print("\nDone. All figures saved to the current directory.")
