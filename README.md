<p align="center">

  <h1 align="center">DROID-W: DROID-SLAM in the Wild</h1>
  <p align="center">
    <a href="https://moyangli00.github.io/"><strong>Moyang Li*</strong></a>
    .
    <a href="https://zzh2000.github.io"><strong>Zihan Zhu*</strong></a>
    .
    <a href="https://people.inf.ethz.ch/pomarc/"><strong>Marc Pollefeys</strong></a>
    .
    <a href="https://cvg.ethz.ch/team/Dr-Daniel-Bela-Barath"><strong>Dániel Béla Baráth</strong></a>
</p>
<p align="center"> <strong>Computer Vision And Pattern Recognition (CVPR) 2026</strong></p>
  <h3 align="center"><a href="https://arxiv.org/abs/2603.19076">Paper</a> | <a href="https://moyangli00.github.io/droid-w/">Project Page</a> | <a href="https://cvg-data.inf.ethz.ch/DROID-W">Dataset</a></h3>
  <div align="center"></div>
</p>
<p align="center">
    <img src="./media/teaser.png" alt="teaser_image" width="100%">
</p>

<p align="center">
Given a casually captured in-the-wild video, DROID-W estimates accurate camera trajectory, scene structure and dynamic uncertainty.
</p>
<br>

<br>
<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#run">Run</a>
    </li>
    <li>
      <a href="#evaluation">Evaluation</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
    <li>
      <a href="#contact">Contact</a>
    </li>
  </ol>
</details>


## Installation

### Requirements
- NVIDIA GPU with CUDA 12.8+ support (tested on RTX 5090 / Blackwell)
- Anaconda or Miniconda
- CUDA Toolkit 12.8+ installed on the system (`nvcc --version` to verify)

### Steps

1. Clone the repo with submodules.
```bash
git clone --recursive https://github.com/rzninvo/DROID-W.git
cd DROID-W
```

2. Create conda environment.
```bash
conda create --name droid-w python=3.11
conda activate droid-w
```

3. Install PyTorch with CUDA 12.8.
```bash
pip install numpy==1.26.3 torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
pip install nvidia-cudnn-cu12 --upgrade
```

4. Install CUDA extensions.
```bash
python -m pip install -e thirdparty/lietorch --no-build-isolation
python -m pip install -e thirdparty/diff-gaussian-rasterization-w-pose --no-build-isolation
python -m pip install -e thirdparty/simple-knn --no-build-isolation
python -m pip install -e . --no-build-isolation
```

5. Install remaining dependencies.
```bash
python -m pip install -r requirements.txt
```

6. Verify installation.
```bash
python -c "import torch; import lietorch; import droid_backends; import diff_gaussian_rasterization; from simple_knn._C import distCUDA2; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

7. Download the pretrained model [droid.pth](https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view?usp=sharing) and place it in the `pretrained` folder.
```bash
mkdir -p pretrained
# Download droid.pth and move it to ./pretrained/droid.pth
```

> **Note:** The `mmcv` package (used by Metric3D) does not currently have pre-built wheels for CUDA 12.8. The code has been patched to make this dependency optional — Metric3D will work without it.

## Run

### Bonn Dynamic Dataset
Download the data as below and the data is saved into the `./Datasets/Bonn` folder. Note that the script only downloads the 8 sequences reported in the paper. To get other sequences, you can download from the [webiste of Bonn Dynamic Dataset](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html).
```bash
bash scripts_downloading/download_bonn.sh
```
You can run DROID-W via the following command:
```bash
python run.py --config ./configs/Dynamic/Bonn/{config_file}
```
We have prepared config files for the 8 sequences. Note that this dataset needs preprocessing the pose. We have implemented that in the dataloader. If you want to test with sequences other than the ones provided, don't forget to specify ```dataset: 'bonn_dynamic'``` in your config file. The easiest way is to inherit from ```bonn_dynamic.yaml```.

### TUM RGB-D (dynamic) Dataset
Download the data (9 dynamic sequences) as below and the data is saved into the `./Datasets/TUM_RGBD` folder. 
```bash
bash scripts_downloading/download_tum.sh
```
The config files for 9 dynamic sequences of this dataset can be found under ```./configs/Dynamic/TUM_RGBD```. You can run DROID as the following:
```bash
python run.py --config ./configs/Dynamic/Wild_SLAM_Mocap/{config_file} 
```

### DyCheck Dataset
Download the [DyCheck](https://drive.google.com/drive/folders/1BHzjHo58nGAMvKMo_AS0_SwU2tJagXXx) dataset and put it in the `./datasets/DyCheck` folder.
The config files for the 12 sequences of this dataset can be found under ```./configs/Dynamic/DyCheck```. You can run DROID-W as the following:
```bash
python run.py --config ./configs/Dynamic/DyCheck/{config_file}
```

### DROID-W Dataset
Download the data (7 dynamic sequences) as below and the data is saved into the `./Datasets/DROID-W` folder. 
```bash
bash scripts_downloading/download_droidw.sh
```
  The config files for 7 dynamic sequences of this dataset can be found under ```./configs/Dynamic/DROIDW```. You can run DROID-W as the following:
```bash
python run.py --config ./configs/Dynamic/DROIDW/{config_file} 
```

### YouTube Sequences
Download the sequences (12 dynamic sequences) as below and the data is saved into the `./Datasets/YouTube` folder.
```bash
bash scripts_downloading/download_youtube.sh
```
  The config files for 12 dynamic sequences of this dataset can be found under ```./configs/Dynamic/YouTube```. You can run DROID-W as the following:
```bash
python run.py --config ./configs/Dynamic/YouTube/{config_file} 
```


## Evaluation

### Camera poses
The camera trajectories will be automatically evaluated after each run of DROID-W except for the DROID-W dataset. Evaluate the camera poses for the DROID-W dataset using the following command:
```bash
python scripts_eval/evaluate_droidw.py
```

We provide a python script to summarize the RMSE of ATE:
```bash
python scripts_eval/summarize_rmse.py -b {path_to_output_dir}
```

## Note

We disable the Gaussian Splatting mapping module by default. If you want to enable it, please set `mapping.enable: True` in `configs/droid_w.yaml`.

## Acknowledgement
We adapted some codes from some awesome repositories including [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM), [WildGS-SLAM](https://github.com/GradientSpaces/WildGS-SLAM), [Metric3D V2](https://github.com/YvanYin/Metric3D), and [DUSt3R](https://github.com/naver/dust3r). Thanks for making codes publicly available. 

## Citation

If you find our code or paper useful, please cite
```bibtex
@inproceedings{Li2026DROIDW,
  author    = {Li, Moyang and Zhu, Zihan and Pollefeys, Marc and Barath, Daniel},
  title     = {DROID-SLAM in the Wild},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## Contact
Contact [Moyang Li](mailto:limoyang2000@gmail.com) for questions, comments and reporting bugs.
