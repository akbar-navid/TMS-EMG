# Motor Cortex Mapping (Forward and Inverse) using TMS-EMG and ConvNets

## Project Summary
Overall System: 

<img src=other/TMS-contribution.png  width="800">

Modeling Pipelines: 

<img src=other/causal-inverse.png width="500">

Inverse Imaging System Network Architecture: 

<img src=https://github.com/neu-spiral/TMS-EMG/assets/38365057/4fb75958-e39a-4a4f-ae28-07917f6fd9a5  width="800">


<!-- ## Prerequisites
Please install all necessary library versions by typing in terminal:

```pip install -r requirements.txt``` -->

## File Structure
```
|--<data>
|--coding file (e.g. M2M-InvNet\train.py)
```

## Usage
An execution sample for training the ```Direct Variational``` model from ```M2M-InvNet```:
```
python train.py --dir <str, data directory> --sub <int, subject number>
```
An execution sample for testing the ```Direct Variational``` model from ```M2M-InvNet```:

```
python test.py --dir <str, data directory> --sub <int, subject number>
```
```test.py``` will generate fold-level statistics (NRMSE, R^2) for the selected subject. Use ```eval.py``` (currently in a Jupyter cell format) for more advanced inference features. Supporting functions are automatically parsed from ```utils.py```. All other models implemented in our paper can be found in ```allModels.py```.

The models from ```M2M-Net``` are in a Jupyter cell format. Please consider running the cells sequentially in VSCode or Spyder.

Matlab scripts used to preprocess and visualize both the forward and inverse models can be found in ```other```.

<!-- ## Usage
Clone this repo, and copy the _\_data_ folder from [here](https:) to the root directory [as shown in the file tree above], for all codes to work.

The code runs from terminal using ```main.py```, with supporting functions automatically parsed from ```models.py```, ```helper.py```, and open-sourced functions from the folder ```extra```.

Plots for results can be generated using ```plot_csv.py```

Some residual code snippets and inline results+visualization can be found in ```multimodal_RA.ipynb```

The raw source files can be found in _/SDrive/CSL/\_Archive/2019/DT\_LONI\_Epileptogenesis\_2019_

![causal-inverse](other/causal-inverse.png)

-->

## Publications
Please take a look at our papers below (and cite if you find helpful), for the corresponding coding folders:

[1] ```M2M-InvNet``` [Inverse Model (EMG->TMS: 2024)](https://ieeexplore.ieee.org/abstract/document/10473158/)

Cite: 
```
@article{akbar2024m2m,
  title={M2M-InvNet: Human Motor Cortex Mapping from Multi-Muscle Response Using TMS and Generative 3D Convolutional Network},
  author={Akbar, Md Navid and Yarossi, Mathew and Rampersad, Sumientra and Lockwood, Kyle and Masoomi, Aria and Tunik, Eugene and Brooks, Dana and Erdo{\u{g}}mu{\c{s}}, Deniz},
  journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  year={2024},
  publisher={IEEE}
}
```

[2] ```M2M-Net``` [Forward Model (TMS->EMG: 2020)](https://dl.acm.org/doi/10.1145/3389189.3389203)

Cite: 
```
@inproceedings{akbar2020m2m,
author = {Akbar, Md Navid and Yarossi, Mathew and Martinez-Gost, Marc and Sommer, Marc A. and Dannhauer, Moritz and Rampersad, Sumientra and Brooks, Dana and Tunik, Eugene and Erdo\u{g}mu\c{s}, Deniz},
title = {Mapping Motor Cortex Stimulation to Muscle Responses: A Deep Neural Network Modeling Approach},
year = {2020},
booktitle = {Proceedings of the 13th ACM International Conference on PErvasive Technologies Related to Assistive Environments},
numpages = {6}
}
```
