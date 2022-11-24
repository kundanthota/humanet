import pickle
import argparse
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle", type=str, required= True, help="path to pickle file")
    parser.add_argument("--gender", type=str, required= True, help="male or female")
    args = parser.parse_args()
    
    pkl = pickle.load(open(args.pickle, 'rb'), encoding='latin1')
    
    try:
        os.mkdir('data')
    except:
        pass

    np.save(os.path.join('data', 'faces.npy'), pkl['f'])
    np.save(os.path.join('data', f'{args.gender}_template.npy'), pkl['v_template'])
    np.save(os.path.join('data', f'{args.gender}_shapedirs.npy'), pkl['shapedirs'])
    

