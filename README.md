# FoMo Dataset Tutorials

Companion notebooks and scripts for the **FoMo (Forêt Montmorency) Multi-Season Robot Navigation Dataset**, hosted on the [AWS Open Data Registry](https://registry.opendata.aws/fomo-dataset).

> Boxan M, Jeanson G, Krawciw A, et al. FoMo: A multi-season dataset for robot navigation in Forêt Montmorency. The International Journal of Robotics Research. 2026;0(0). doi:10.1177/02783649261491481

---

## About the Dataset

The FoMo dataset is a year-long robotic data collection recorded in a boreal forest 80 km north of Quebec City, Canada. It features **over 64 km of trajectories** repeated across **12 deployments** spanning dramatic seasonal changes — from −19 °C winters with over 1 m of snow to warm summers with dense vegetation.

**Sensors:** 2× lidar, 1× FMCW radar, stereo camera, monocular camera, 2× IMU, 2× microphone, GNSS  
**Ground truth:** PPK-GNSS with per-point covariances, TUM format  
**Storage:** `s3://fomo-dataset` (public, no AWS credentials required)  
**Website:** [fomo.norlab.ulaval.ca](https://fomo.norlab.ulaval.ca)

---

## Contents

```
.
├── get_to_know_fomo-dataset.ipynb   # Guided tour notebook (Jupyter)
├── get_to_know_fomo-dataset.py      # Same tutorial as a standalone Python script
└── README.md                # This file
```

---

## Get to Know the Dataset

The main tutorial — `fomo_get_to_know.ipynb` — is a guided tour of the dataset that answers six key questions:

| Section | Description |
|---------|-------------|
| **Q1** | How is the dataset organized? |
| **Q2** | What data formats are present? |
| **Q3** | How do I download and load data? (CSV, IMU, lidar, radar, ground truth, calibration) |
| **Q4** | Visualizations — trajectory, lidar BEV, radar polar image, seasonal conditions, power consumption |
| **Q5** | One answered question: robustness of localization methods to seasonal changes |
| **Q6** | One open challenge: terrain traversability prediction across seasons |

---

## Quick Start

### Requirements

```bash
pip install boto3 pandas numpy matplotlib Pillow
```

No AWS account is required — the dataset is publicly accessible.

### Run as a Jupyter notebook

```bash
jupyter notebook fomo_get_to_know.ipynb
```

### Run as a Python script

```bash
python fomo_get_to_know.py
```

---

## License

The FoMo dataset is publicly available under the license described in `s3://fomo-dataset/LICENSE.txt`.

---

## Citation

```bibtex
@article{doi:10.1177/02783649261491481,
    author = {Matěj Boxan and Gabriel Jeanson and Alexander Krawciw and Effie Daum and Xinyuan Qiao and Sven Lilge and Timothy D Barfoot and François Pomerleau},
    title ={FoMo: A multi-season dataset for robot navigation in Forêt Montmorency},
    journal = {The International Journal of Robotics Research},
    volume = {0},
    number = {0},
    pages = {02783649261491481},
    year = {2026},
    doi = {10.1177/02783649261491481},
    URL = {https://doi.org/10.1177/02783649261491481},
    eprint = {https://doi.org/10.1177/02783649261491481},
}
```
