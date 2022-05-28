import numpy as np
import json
from PIL import Image 
import trimesh
from shutil import copyfile
import os
import pickle
import argparse

def args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--resolution', type = int, required = True,\
                        help='for 300x300 image enter 300')
                        
    parser.add_argument('--gender', type = str, required = True,\
                        help='male or female')

    parser.add_argument('--loader_type', type = str, required = True,\
                        help='train or test')
    
    arguments = parser.parse_args()
    return arguments

def import_data(gender, loader_type):
    """
    copies data from calvis folder to the data folder created.

    create a train_test seperated json file in the following format
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

    h_w_measures : run utils/measures.py --path /path/to/the/obj/files --gender male/female.
    """
    
    data = json.load(open('data/train_test_data.json', 'r'))
    for subject in data[gender][loader_type]:
        obj_file = f"CALVIS/dataset/cmu/human_body_meshes/{gender}/subject_mesh_{subject}.obj"
        vertices = trimesh.load(obj_file).vertices
        np.save(f"data/{gender}/{subject}/vertices.npy", vertices)

        measures = f"CALVIS/dataset/cmu/annotations/{gender}/subject_mesh_{subject}_anno.json"
        destination = f"data/{gender}/{subject}/measures.json"

        copyfile(measures, destination)

def save_dataloaders(gender, resolution, loader_type):
    """
    creates and save dataloaders (train/test) 
    """
    data = json.load(open('data/train_test_data.json', 'r'))
    h_w_m = json.load(open('data/h_w_measures.json', 'r'))
    front = []
    side = []
    h_w_measures = []
    measures = []
    vertices = []
    betas = []
    for subject in data[gender][loader_type]:
        
        front_path = f"data/{gender}/{subject}/{resolution}/front.png"
        side_path = f"data/{gender}/{subject}/{resolution}/right.png"
        
        measures_path = f"data/{gender}/{subject}/measures.json"

        vertices_path = f"data/{gender}/{subject}/vertices.npy"
        
        front.append(np.array(Image.open(front_path).convert("L")))
        side.append(np.array(Image.open(side_path).convert("L")))

        h_w_measures.append(np.array(h_w_m[gender][subject]))

        m = json.load(open(measures_path, "r"))

        measures.append(np.array([m["human_dimensions"]["chest_circumference"],\
         m["human_dimensions"]["pelvis_circumference"], m["human_dimensions"]["waist_circumference"]]))

        betas.append(np.array(m['betas']))

        vertices.append(np.load(vertices_path))

    front = np.expand_dims(np.array(front), axis = -1)
    side = np.expand_dims(np.array(side), axis = -1)

    images = np.concatenate([front, side], axis = -1)
    measures = np.array(measures)
    vertices = np.array(vertices)
    h_w_measures = np.array(h_w_measures)
    betas = np.array(betas)


    np.save(f"data/dataloaders/{gender}/{loader_type}_{resolution}_images.npy", images)
    np.save(f"data/dataloaders/{gender}/{loader_type}_vertices.npy", vertices)
    np.save(f"data/dataloaders/{gender}/{loader_type}_h_w_measures.npy", h_w_measures)
    np.save(f"data/dataloaders/{gender}/{loader_type}_measures.npy", measures)
    np.save(f"data/dataloaders/{gender}/{loader_type}_betas.npy", betas)

def get_smpl_data(gender):
    """
    loads smpl template and save principal shapes, and template 
    """
    smpl_path = f"SMPL/{gender}_template.pkl"
    pkl = pickle.load(open(smpl_path, 'rb'), encoding= 'latin1')
    np.save(f"data/{gender}_shapedirs.npy", pkl["shapedirs"])
    np.save("data/faces.npy", pkl["f"])
    np.save(f"data/{gender}_template.npy", pkl["v_template"])

def main():
    arguments = args()
    gender = arguments.gender
    resolution = arguments.resolution
    loader_type = arguments.loader_type
    try:
        os.mkdir("data/dataloaders")
    except:
        pass

    try:
        os.mkdir(f"data/dataloaders/{gender}")
    except:
        pass

    import_data(gender, loader_type)
    get_smpl_data(gender)
    save_dataloaders(gender, resolution, loader_type)
    
if __name__ == "__main__":
    main()

