from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse
import torch
import numpy as np
from joblib import load
from sklearn.decomposition import PCA
from measurement_evaluator import Human
from utils.img2mask import Img2Mask
from utils.image_utils import ImgSizer
from utils.model import *
from utils.torchloader import *
import matplotlib.pyplot as plt
import trimesh

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--experiment", type=str, required = True, help='Give experiment name')
    parser.add_argument("--front", type=str, required= True, help="path to front view image")
    parser.add_argument("--side", type=str, required= True, help="path to side view image")
    parser.add_argument("--gender", type=str, required = True)
    parser.add_argument("--height", type=float, required = True)
    parser.add_argument("--weight", type=float, required = True)
    parser.add_argument("--feature_model", type=str, default='ae')
    parser.add_argument("--mesh_name", type=str, default='subject.obj')
    parser.add_argument("--measurement_model", type=str, default='calvis')

    args = parser.parse_args()

    try:
        os.mkdir(os.path.join('data', 'demo'))
    except:
        pass
    
    try:
        os.mkdir(os.path.join('data/demo', args.experiment))
    except:
        pass

    
    front = np.array(Image.open(args.front).convert('L'))/255.0
    side = np.array(Image.open(args.side).convert('L'))/255.0
    
    print("Data preprocessed! \n Extracting Important features")

    if args.feature_model == 'ae':
        
        feature_extractor_path = f'weights/feature_extractor_{args.gender}_50.pth'
        feature_extractor_weights = torch.load(feature_extractor_path,  map_location=torch.device('cpu'))
        feature_extractor = Deep2DEncoder(image_size= 512 , kernel_size=3, n_filters=32)
        feature_extractor.load_state_dict(feature_extractor_weights)
        
        feature_extractor.eval()
        feature_extractor.requires_grad_(False)

        front = torch.tensor(front).float()
        side = torch.tensor(side).float()

        front_features = feature_extractor(front.view(1, 1, 512, 512)).detach().numpy().reshape(256)
        side_features = feature_extractor(side.view(1, 1, 512, 512)).detach().numpy().reshape(256)
        
        front_features = np.array(front_features)
        side_features = np.array(side_features)

        features = np.concatenate([front_features, side_features], axis = -1).reshape(1,512)
    
    if args.feature_model == 'pca':
        
        pca_front = load(f'weights/pca_{args.gender}_front.joblib')
        pca_side = load(f'weights/pca_{args.gender}_side.joblib')
        
        front_features = pca_front.transform(np.array(front).reshape(1, 512*512))
        side_features = pca_side.transform(np.array(side).reshape(1, 512*512))

        front_features = np.array(front_features)
        side_features = np.array(side_features)

        features = np.concatenate([front_features, side_features], axis = -1)
    print("Feature Extraction done \n Estimating Measurements")
    template = np.load(f'data/{args.gender}_template.npy')
    shape_dirs = np.load(f'data/{args.gender}_shapedirs.npy')
    faces =  np.load(f'data/faces.npy')
    
    if args.measurement_model == 'nomo':
        features = np.concatenate([features, np.array(args.height).reshape(1,1)], axis = -1)
    else:
        features = np.concatenate([features, np.array(args.height).reshape(1,1), np.array(args.weight).reshape(1,1)], axis = -1)

    
    if args.measurement_model == 'nomo':
        human = load(f'weights/nomo_{args.gender}_krr.pkl')
    
    else:
        human = load(f'weights/calvis_{args.gender}_krr.pkl')
    
    measurements = human.predict_measurements(features)
    shape = human.predict_shape(features)

    print(f"Chest Circumference : {measurements[0][0]}")
    print(f"Hip Circumference : {measurements[0][1]}")
    print(f"Waist Circumference : {measurements[0][2]}")

    mesh = human.display_3D(shape)
    mesh.export(os.path.join(f'data/demo/{args.experiment}',args.mesh_name))
    print("3D model saved!")


if __name__ == "__main__":
    main()

   


    




    


    

    


    