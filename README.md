<div style="text-align: center; margin: 10px">
    <h1> ⭐ <span style='color: #4D96FF;'>M</span><span style='color: #F9D923;'>I</span><span style='color: #EB5353;'>S</span>:
        <span style='color: #4D96FF;'>M</span>ulti-granularity <span style='color: #F9D923;'>I</span>nteraction <span style='color: #EB5353;'>S</span>imulation for Unsupervised Interactive Segmentation </h1>
</div>
<p align="center">
    <a href="https://arxiv.org/pdf/2303.13399.pdf"><img src="http://img.shields.io/badge/Paper-EB455F.svg?logo=arxiv" style="display:inline;"></a>
    <a href="https://lkhl.github.io/MIS"><img src="http://img.shields.io/badge/Project_Page-7149C6.svg?logo=openproject" style="display:inline;"></a>
    <a href="https://lkhl.github.io/MIS"><img src="https://img.shields.io/badge/Video-FC2947.svg?logo=youtube" style="display:inline;"></a>
    <a href="https://huggingface.co/spaces/lkhl/MIS?duplicate=true"><img src="https://img.shields.io/badge/Demo-EA906C?logo=buffer" style="display:inline;"></a>
</p>
This is an official implementation for our ICCV'23 paper "Multi-granularity Interaction Simulation for Unsupervised Interactive Segmentation".

![](https://lkhl.github.io/MIS/static/images/method_2.png)

## Installation

### **Environment**

**Step 1: setup python environment**

```shell
# clone this repository
git clone https://github.com/lkhl/MIS
cd MIS

# create conda environment
conda create -n mis python=3.9
pip install -r requirements.txt
```

**Step 2: install other dependencies**

- CMake

```shell
sudo apt-get install cmake
```

- Eigen

If you have already installed eigen (3.4.0 is suggested) on your machine, please ignore this step.

```shell
# download the source code
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar
tar -xf eigen-3.4.0.tar
cd eigen-3.4.0

# install eigen
mkdir build
cd build
cmake ..
sudo make install
```

**Step 3: build C++ extensions**

```shell
cd mis/ops
bash install.sh
```

### Dataset

Please follow [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation) to prepare the GrabCut, Berkeley, SBD, and DAVIS datasets.

## Usage

### Evaluate

```shell
apython evaluate_model.py NoBRS \
	--checkpoint /path/to/checkpoint \
	--datasets GrabCut,Berkeley,SBD,DAVIS
```

The results and pre-trained model are as follows

<table>
    <thead align="center">
        <tr>
            <th rowspan="2">Model</th>
            <th colspan="2">GrabCut</th>
            <th colspan="2">Berkeley</th>
            <th colspan="2">SBD</th>
            <th colspan="2">DAVIS</th>
        </tr>
        <tr>
            <td>NoC@85</td>
            <td>NoC@90</td>
            <td>NoC@85</td>
            <td>NoC@90</td>
            <td>NoC@85</td>
            <td>NoC@90</td>
            <td>NoC@85</td>
            <td>NoC@90</td>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td align="left"><a href="https://">ViT-Base</a></td>
            <td>1.94</td>
            <td>2.32</td>
            <td>3.09</td>
            <td>4.58</td>
            <td>6.91</td>
            <td>9.51</td>
            <td>6.33</td>
            <td>8.44</td>
        </tr>
    </tbody>
</table>

### Demo

We provide a demo for showing the merging process and the interactive segmentation results based on [gradio](https://www.gradio.app/), which can be launched by

```shell
python app.py
```

### Training

**Step 1: preprocessing**

```shell
python preprocess.py -d /path/to/SBD

optional arguments:
  -h, --help            show help message and exit
  --data-root DATA_ROOT, -d DATA_ROOT
                        Root directory for the SBD dataset
  --out-dir OUT_DIR, -o OUT_DIR
                        Output directory for the preprocessed data
  --model-size {small,base,large,giant}, -m {small,base,large,giant}
                        Model size of the ViT
  --patch-size {8,14,16}, -p {8,14,16}
                        Patch size of the ViT
  --n-featurizing-workers N_FEATURIZING_WORKERS
                        Number of workers for featurizing. Set to 0 to disable parallel processing
  --n-merging-workers N_MERGING_WORKERS
                        Number of workers for merging. Set to 0 to disable parallel processing
```

The processed data will be saved in `./data/proposals/sbd` by default.

**Step 2: training**

Use the following command to train a model based on SimpleClick with randomly sampled proposals.

```shell
python train.py models/mis_simpleclick_base448_sbd.py
```

## Acknowledgements

This repository is built upon [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation) and [SimpleClick](https://github.com/uncbiag/SimpleClick). The project page is built using the template of [Nerfies](https://nerfies.github.io/). Thank the authors of these open source repositories for their efforts. And thank the ACs and reviewers for their effort when dealing with our paper.

## Citing

If you find this repository helpful, please consider citing our paper.

```
@article{li2023multi,
  title={Multi-granularity interaction simulation for unsupervised interactive segmentation},
  author={Li, Kehan and Zhao, Yian and Wang, Zhennan and Cheng, Zesen and Jin, Peng and Ji, Xiangyang and Yuan, Li and Liu, Chang and Chen, Jie},
  journal={arXiv preprint arXiv:2303.13399},
  year={2023}
}
```
