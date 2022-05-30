
## Humanet for 3D Shape and Clothing Measurements


pytorch implementation of the humanet, an auto-encoder, for **human shape space** with pose-independent, introduced in the CVPR 2020 paper:

**Estimation of 3D Body Shape and Clothing Measurements from Frontal- and Side-view Images** <br>
Kundan Thota, Sungho Suh, Bo Zhou, Paul Lukowicz<br>
[Full paper]()

## Installation

We recommend creating a new virtual environment for a clean installation of the dependencies. All following commands are assumed to be executed within this virtual environment. The code has been tested on Ubuntu 18.04, python 3.6 and CUDA 10.0.

```bash
python3 -m venv humanenv
source humanenv/bin/activate
pip install -U pip setuptools
```

- `pip install -r requirements.txt`
- Download the [SMPL body model](https://smpl.is.tue.mpg.de/) (Note: use the version 1.0.0 with 10 shape PCs), and place the renamed `{gender}_template.pkl` files for both genders and put them in `./smpl/`.
## Quick demo 

- Download the SMPL body model as described above and our [pre-trained demo model]() put the downloaded folder under the `weights` folder:

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

Then run:

```bash
python demo.py --front_img /path/to/512x512/front/image --side_img /path/to/512x512/side/image --gender male/female \
                     --height (in cms) --weight (in kilos) --mesh_name /name/for/the/model.obj  
```

It will generate clothing measurements and the 3D model.

## Process data, training and evaluation
### Prepare training data
Here we assume that the [CALVIS dataset](https://github.com/neoglez/calvis) is downloaded. The dataset is placed in the ./project/folder. The front and the side images of the 3D human is captured as a scene and stored under ./data folder
```bash
python capture_images.py --resolution 512 --gender male/female --path path/to/.obj files/in/CALVIS/folder
```
will create scene images under data folder.

    - create a train_test seperated json file in the following format
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
python utils/measures.py --path /path/to/the/obj/files --gender male/female.
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

**Clothing Measurement Error**:

|       | male dataset      | female dataset        |
| -----------:|---------:|------------:|
| chest   | 5.21 ± 5.23 | 3.37 ± 7.67 |
| -----------:|---------:|------------:|
| waist   | 2.28 ± 2.66 | 2.29 ± 2.36 | 
| -----------:|---------:|------------:|
| hip   | 2.8 ± 2.66 | 2.75 ± 2.61 | 

**3D shape Error**:
|       | male dataset      | female dataset        |
| per vertex error   | 0.52 ± 1.01 | 0.48 ± 0.94 | 
