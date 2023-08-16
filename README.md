# Introduction
The source code and models for our paper **Combating Noisy Labels by Partial Label Learning**
# Framework
![Our Framework](https://github.com/LPL-LPL/LPL/blob/main/framework.png)
# Installation
After creating a virtual environment of python 3.7, run `pip install -r requirements.txt` to install all dependencies
# How to use
The code is currently tested only on GPU.
* Data preparation  
  Created a folder `Datasets` and download `cifar100`/`web-aircraft`/`web-bird`/`web-car`/`Animal10N`/`mini-webvision` dataset into this folder.
* Source code
  * If you want to train the whole model from beginning using the source code, please follow subsequent steps:
    *  Prepare data
    *  Modify GPU device in the corresponding train script `xxx.sh` in `scripts` folder
    *  Activate virtual environment (e.g. conda) and then run    
      `bash scripts/xxx.sh` 
