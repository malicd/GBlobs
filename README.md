# [RoboSense 2025 Challenge - Track 3: Sensor Placement] LRP Solution

This repository contains the official code and pre-trained checkpoints for our submission to the RoboSense 2025 Challenge, Track 3: Sensor Placement. Our approach is built upon [**GBlobs**](https://arxiv.org/abs/2503.08639), a parameter-free method designed to explicitly capture the local geometric structure within point clouds.

## Methodology
Our method leverages the domain generalization capabilities of GBlobs, which significantly improves performance, particularly in the near range where point density is sufficient for high-quality local feature estimation. In the far range, where local features may degrade due to sparse point density, we found that relying on global coordinates yields better empirical results.

This dual-strategy approach allows us to combine the strengths of both methods: local GBlobs features for near-field accuracy and global coordinates for robust far-field performance.

## Environment Setup

To reproduce our results, you need to clone the repository, set up the virtual environment, and download the checkpoints hosted on [HuggingFace](https://huggingface.co/dmalic/RoboSense25_Track_3). The following steps will guide you through the process.

### Clone the repository
First, clone the specific branch for this challenge:
```bash
git clone -b robosense25_track3 git@github.com:malicd/GBlobs.git robosense25_track3_gblobs
cd robosense25_track3_gblobs
```

### Setup the Environment
Next, create and activate a Python virtual environment, then install the necessary dependencies:
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install spconv-cu118
pip install -r requirements.txt
python setup.py develop
```

### Get the Checkpoints
The model checkpoints are hosted on HuggingFace. Use Git LFS to clone the repository and download the files:
```bash
git lfs install
git clone https://huggingface.co/dmalic/RoboSense25_Track_3
```

## Reproducing the Results

Follow these steps to generate the results and reproduce our final submission.

Navigate to the tools directory. All necessary scripts are located in the tools directory:
```bash
cd tools
```

### Get GBlobs estimates
Run the following command to get the GBlobs-based predictions. Make sure to note the output directory, as you'll need it later.
```bash
python test.py --cfg_file cfgs/robosense_models/transfusion_lidar_gblobs.yaml  --batch_size 1 --ckpt ../RoboSense25_Track_3/transfusion_lidar_gblobs/default/ckpt/checkpoint_epoch_90.pth
```
This will save the results to ../output/<path to Gblobs output>/results_nusc.json.

### Generate Global Estimates
Run the next command to get the global coordinate-based predictions, also noting the output path.
```bash
python test.py --cfg_file cfgs/robosense_models/transfusion_lidar.yaml  --batch_size 1 --ckpt ../RoboSense25_Track_3/transfusion_lidar/default/ckpt/checkpoint_epoch_90.pth
```
This will save the results to ../output/<path to global output>/results_nusc.json.

### Merge the Results

Finally, use the `ensamble.py` script to merge the two sets of predictions. The script combines the GBlobs estimates for objects within 30 meters and uses the global estimates for objects beyond this threshold.

```bash
python ensamble.py --path_gblobs ../output/<path to Gblobs output>/results_nusc.json  --path_glob ../output/<path to global output>/results_nusc.json --thold 30
```

---

## Dataset Preparation

First, download the dataset from the [official challenge website](https://robosense2025.github.io/track3).
To prepare the dataset, run the following command to create the test info files.

```bash
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/robosense_dataset.yaml --version v1.0-test
```

## Train the models
We trained our models using 4x NVIDIA RTX A6000 50GB GPUs. The following commands will train both the GBlobs and global versions.
```bash
./scripts/torch_train.sh 4 --cfg_file cfgs/robosense_models/transfusion_lidar_gblobs.yaml
./scripts/torch_train.sh 4 --cfg_file cfgs/robosense_models/transfusion_lidar.yaml
```
