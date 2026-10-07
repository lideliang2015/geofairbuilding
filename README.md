# GeoFair-Building v1.0

Reconstruction scripts and documentation for geospatial fairness evaluation in building extraction.

This repository accompanies the manuscript:

> **The GeoFair Framework: A Spatially Explicit Multi-Scale Approach for Quantifying and Mitigating Geographic Unfairness in Building Extraction**  
> Deliang Li, Tao Liu, Shuangtong Li, Rongsheng Fan, Pan Li  


---

## Overview

GeoFair-Building v1.0 is a hierarchically stratified benchmark for evaluating geographic fairness in deep learning-based building extraction. It covers six continents and three Human Development Index (HDI) levels, and is built entirely from publicly available datasets.

The repository provides scripts for:

- Data preprocessing and integration
- Model training and evaluation (SegFormer, U-Net, DeepLabV3+, SAM)
- Fairness metrics (SFR, SFI, Getis-Ord Gi*)
- Diagnostic analysis (geographical detector, MGWR)
- Mitigation experiments (geographically weighted loss, etc.)

---

## Data Sources

The benchmark integrates the following public datasets:

| Dataset | Source |
|---|---|
| WHU Building Dataset | http://gpcv.whu.edu.cn/data/building_dataset.html |
| Inria Aerial Image Labeling | https://project.inria.fr/aerialimagelabeling |
| SpaceNet | https://spacenet.ai/datasets/ |
| WHU_Satellite_II | http://gpcv.whu.edu.cn/data/building_dataset.html |

Due to license restrictions, the integrated dataset is **not redistributed**. Researchers can reproduce the test set by following the provided scripts and documentation.

---

## Repository Structure
geofairbuilding/
├── config.py # Configuration and paths
├── requirements.txt # Python dependencies
├── data_preprocess/ # Data preprocessing scripts
├── dataset.py # Dataset loader
├── debug_metadata.py # Metadata verification
├── models.py # Model definitions
├── evaluate_fairness.py # Fairness metrics (SFR, SFI, Gi*)
├── evaluate_segformer.py # SegFormer evaluation
├── evaluate_sam.py # SAM evaluation
├── reevaluate_unet.py # U-Net re-evaluation
└── README.md


---

## Installation

1.Clone the repository:

```bash
git clone https://github.com/lideliang2015/geofairbuilding.git
cd geofairbuilding
```



2.Create a virtual environment (recommended):

```bash
conda create -n geofair python=3.9
conda activate geofair
```

3.Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

1.Download the original datasets from the links above.
2.Organize them following the structure expected by data_preprocess/.
3.Run the preprocessing scripts to generate the GeoFair-Building v1.0 test set.

## Example:
```bash
python data_preprocess/build_geofair.py
```
## Usage
## Fairness Evaluation

```bash
python evaluate_fairness.py --pred_dir ./predictions --meta ./metadata.csv
```
Outputs: SFR, SFI, Getis-Ord Gi* hotspots.

## Model Evaluation
# SegFormer:
```bash
python evaluate_segformer.py
```

# SAM (zero-shot):

```bash

python evaluate_sam.py
```

# U-Net:

```bash

python reevaluate_unet.py
```

## Diagnostic Analysis
Geographical detector and MGWR are implemented in evaluate_fairness.py and can be run separately.

## Results
Key findings from the manuscript:

Spatial Fairness Ratio (SFR) = 0.579
Best region: Oceania (IoU = 0.659)
Worst region: South America (IoU = 0.382)
Dominant driver: Training sample density (q = 0.683)
GWL improves SFR to 0.641 with only 1.2 pp drop in global IoU

## Citation
If you use this repository or the GeoFair-Building v1.0 benchmark, please cite:

@article{li2026geofair,
  title={The GeoFair Framework: A Spatially Explicit Multi-Scale Approach for Quantifying and Mitigating Geographic Unfairness in Building Extraction},
  author={Li, Deliang and Liu, Tao and Li, Shuangtong and Fan, Rongsheng and Li, Pan},
  journal={Transactions in GIS},
  year={2026},
  note={Under review}
}

## License
The code in this repository is released under the MIT License.
The original datasets remain under their respective licenses.

## Contact
For questions, please contact:

Deliang Li: 108595403@qq.com
