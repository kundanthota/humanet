
## Humanet for 3D Shape and Clothing Measurements


pytorch implementation of the humanet, an auto-encoder, for **human shape space** with pose-independent, introduced:

**Estimation of 3D Body Shape and Clothing Measurements from Frontal- and Side-view Images** <br>
Kundan Thota, Sungho Suh, Bo Zhou, Paul Lukowicz<br>
[Full paper](https://arxiv.org/pdf/2205.14347.pdf)

## Installation

We recommend creating a new virtual environment for a clean installation of the dependencies. All following commands are assumed to be executed within this virtual environment. The code has been tested on Scientific Linux 7.9, python 3.10 and CUDA 10.1.

```bash
python3 -m venv humanenv
source humanenv/bin/activate
pip install -U pip setuptools
```

- `pip install -r requirements.txt`
- Download the [SMPL body model](https://smpl.is.tue.mpg.de/) (Note: use the version 1.0.0 with 10 shape PCs), and place the renamed `{gender}_template.pkl` files for both genders and put them in `./smpl/`.
## Quick demo 

- Download the SMPL body model as described above and our [pre-trained demo model](https://drive.google.com/file/d/1BRjwWPn085pAsKRTYa2EyBQQ3SjaiU6U/view?usp=sharing) put the downloaded folder under the `weights` folder:

```
    humanet
    ├── CALVIS
    |   └── ...
    ├── data                             # Folder with preprocessed data
    |   └── ...
    ├── weights
    |   ├── feature_extractor_female_50.pth                
    │   ├── feature_extractor_male_50.pth  
    |   ├── calvis_female_krr.pkl                
    │   ├── calvis_male_krr.pkl  
    ├── smpl
    |   ├── female_template.pkl                  
    │   ├── male_template.pkl               
    └── ...
```

Then steps to run for a demo:

### step-1: 

Run the following snippet to initially create SMPL starter files.

```bash
python utils/preprocess_smpl.py --pickle /path/to/gender_pickle/file --gender male/female
```
### step-2: 

Note: Please try to square crop humans perfectly to fit in the image without additional objects for the better results as shown in the paper. Run following command to resize the RGB images to 512x512 resolution images.

```bash
python utils/image_utils.py --front /path/to/front/image --side /path/to/side/image
```
### step-3:

please follow the following notebook to segment the images created from step-2.

Jupyter notebook with the example: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZHQ3beJP-7Pbq4I5Jsc8Co2dIkK31ALi?usp=sharing)

### step-4:

Run the following command to produce a demo. Note: Please input the images in same angles as used in the paper.

```bash
python demo.py --front_img /path/to/512x512/front/image/from/step-3 --side_img /path/to/512x512/side/image/from/step-3  --gender male/female \
                     --height (in meters) --weight (in kilos) --mesh_name /name/for/the/model.obj  
```

It will generate clothing measurements and the 3D model.

## Process data, training and evaluation
### Prepare training data
Here we assume that the [CALVIS dataset](https://github.com/neoglez/calvis) is downloaded. The dataset is placed in the ./project/folder. The front and the side images of the 3D human is captured as a scene and stored under ./data folder
```bash
python capture_images.py --resolution 512 --gender male/female --path path/to/.obj files/in/CALVIS/folder
```
will create scene images under data folder.

    - create a train_test with 80/20 split json file in the following format
                  {
                      male:{
                          train:[sub_id1, sub_id2, ...],
                          test:[sub_id1, sub_id2, ...]
                      },
                      female:{
                          train:[sub_id1, sub_id2, ...],
                          test:[sub_id1, sub_id2, ...]
                      }
                  }.
    
```bash
python utils/measures_.py --path /path/to/the/obj/files --gender male/female.
```

### Training

Once the data is organized, we are ready for training:

```bash
python trainer.py --data_path /path/to/trainloader.npy --gender male/female --loss bce
```
 
The training will start. To customize the training, check the arguments defined in `trainer.py`, and set them accordingly.


### Evaluation

Once the training is done, extract the low embedding space of the humans by running the following:

```bash
python evaluator.py --data_path /path/to/dataloader.npy --gender male/female --mode features 
```

Once the features are extracted run the following command to check the results:

```bash
python measurement_evaluator.py --gender female/male
```

### Performance

**Clothing Measurement Error** (in mm):

|       | male dataset | female dataset |
|:-----:|:------------:|:--------------:|
| chest |  5.21 ± 5.23 |   3.37 ± 7.67  |
| waist |  2.28 ± 2.66 |   2.29 ± 2.36  |
|  hip  |  2.8 ± 2.66  |   2.75 ± 2.61  |

**3D shape Error** (in milli-units):
|                  | male dataset | female dataset |
|:----------------:|:------------:|:--------------:|
| per vertex error |  0.52 ± 1.01 |   0.48 ± 0.94  | 

## Citations

If you wanna use our code/work in your reasearch, please consider citing:

```bash

@INPROCEEDINGS{9897520,  
author={Prabhu Thota, Kundan Sai and Suh, Sungho and Zhou, Bo and Lukowicz, Paul},  
booktitle={2022 IEEE International Conference on Image Processing (ICIP)},   
title={Estimation Of 3d Body Shape And Clothing Measurements From Frontal-And Side-View Images},   
year={2022},  
volume={},  
number={},  
pages={2631-2635},  
doi={10.1109/ICIP46576.2022.9897520}}

```
