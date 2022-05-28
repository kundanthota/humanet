import argparse
import os
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils.torchloader import *
from utils.model import *
import cv2
import warnings
from time import sleep
import time
from utils.losses import MAASE, Accuracy
from sklearn.decomposition import PCA
from joblib import load
from sklearn.kernel_ridge import KernelRidge

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

class PCAExtractor(object):
    def __init__(self, gender, loss):
        self.pca_front = load(f'weights/pca_{gender}_front.joblib')
        self.pca_side = load(f'weights/pca_{gender}_side.joblib')
        
        losses = {'mse' : nn.MSELoss(),\
                  'mae' : nn.L1Loss(),\
                  'mae+mse': MAASE(),\
                  'bce': nn.BCELoss()}
        self.loss = losses[loss]
        self.accuracy = Accuracy()

    def test_performance(self,data):
        front, side = data[:,:,:,0], data[:,:,:,1]
        side_data = side.reshape(data.shape[0], data.shape[1]*data.shape[2])
        front_data = front.reshape(data.shape[0], data.shape[1]*data.shape[2])

        start_time = time.time()

        front_features = self.pca_front.transform(front_data)
        side_features = self.pca_side.transform(side_data)
        recon_side_data = self.pca_side.inverse_transform(side_features)
        recon_front_data = self.pca_side.inverse_transform(front_features)
        recon_side_data = recon_side_data.reshape(len(data), 512,512)
        recon_front_data = recon_front_data.reshape(len(data), 512,512)

        front = torch.tensor(front).float()
        side = torch.tensor(side).float()
        recon_front_data = torch.tensor(recon_front_data).float()
        recon_side_data = torch.tensor(recon_side_data).float()
        
        front_loss = self.loss(front, recon_front_data).item()
        side_loss = self.loss(side, recon_side_data).item()
        front_accuracy = self.accuracy(front, recon_front_data).item()
        side_accuracy = self.accuracy(side, recon_side_data).item()

        elapsed_time = time.time() - start_time
        print('Time={:.2f}s test_loss_front={:.4f} test_loss_side={:.4f} accuracy_front= {:.4f} accuracy_side= {:.4f}'.format(
                                elapsed_time, 
                                front_loss, side_loss, front_accuracy, side_accuracy))    
 
    def extract_features(self,data):
        front_data, side_data = data[:,:,:,0], data[:,:,:,1]
        side_data = side_data.reshape(data.shape[0], data.shape[1]*data.shape[2])
        front_data = front_data.reshape(data.shape[0], data.shape[1]*data.shape[2])
        front_features = self.pca_front.transform(front_data)
        side_features = self.pca_side.transform(side_data)
        features = np.concatenate([front_features, side_features], axis = -1)
        return features


class AEXtractor(object):
    def __init__(self, device, gender, batch_size, loss):
        self.device = device
        self.gender = gender
        self.batch_size = batch_size

        losses = {'mse' : nn.MSELoss(),\
                  'mae' : nn.L1Loss(),\
                  'mae+mse': MAASE(),\
                  'bce': nn.BCELoss()}
        
        self.loss = losses[loss]
        self.accuracy = Accuracy()

        feature_extractor_path = f'weights/feature_extractor_{self.gender}_50.pth'
        feature_extractor_weights = torch.load(feature_extractor_path,  map_location=torch.device(self.device))
        feature_extractor = Deep2DEncoder(image_size= 512 , kernel_size=3, n_filters=32)
        feature_extractor.load_state_dict(feature_extractor_weights)

        decoder_path = f'weights/decoder_{self.gender}_50.pth'
        decoder_weights = torch.load(decoder_path,  map_location=torch.device(self.device))
        decoder = Deep2DDecoder(image_size= 512 , kernel_size=3, n_filters=32)
        decoder.load_state_dict(decoder_weights)

        self.feature_extractor = feature_extractor.to(self.device)
        self.decoder = decoder.to(self.device)

    def test_performance(self, data_loader):

        feature_extractor = self.feature_extractor
        decoder = self.decoder

        reconstruction_criterion = self.loss
    
        epoch_loss_front= 0
        epoch_loss_side = 0
        total_accuracy_front = 0
        total_accuracy_side = 0
        
        feature_extractor.eval()
        decoder.eval()
        feature_extractor.requires_grad_(False)
        decoder.requires_grad_(False)
        
        step=0
        start_time = time.time()
        
        for x_front, x_side in data_loader:
            x_front = x_front.to(device=self.device)
            x_side = x_side.to(device=self.device)
            feature_front = feature_extractor(x_front)
            feature_side = feature_extractor(x_side)

            x_hat_front = decoder(feature_front)
            x_hat_side = decoder(feature_side)
            
            total_accuracy_front += self.accuracy(x_hat_front, x_front)
            total_accuracy_side += self.accuracy(x_hat_side, x_side)

            recon_loss_front = reconstruction_criterion(x_hat_front, x_front)
            recon_loss_side = reconstruction_criterion(x_hat_side, x_side)
            recon_loss = recon_loss_front+recon_loss_side

            epoch_loss_front += recon_loss_front.item()
            epoch_loss_side += recon_loss_side.item()
            
            step+=1
        
        epoch_loss_front = epoch_loss_front / step
        epoch_loss_side = epoch_loss_side / step
        total_accuracy_front = total_accuracy_front / step
        total_accuracy_side = total_accuracy_side / step

        elapsed_time = time.time() - start_time 
        print('Time={:.2f}s test_loss_front={:.4f} test_loss_side={:.4f} accuracy_front= {:.4f} accuracy_side= {:.4f}'.format(
                                elapsed_time,
                                epoch_loss_front, epoch_loss_side, total_accuracy_front, total_accuracy_side))
        
    
    def extract_features(self, data):
        
        data = torch.tensor(data).float()
        front = data[:, :, :, 0]
        side = data[:, :, :, 1]

        front_features = []
        side_features = []
        
        self.feature_extractor.eval()
        self.feature_extractor.requires_grad_(False)

        for f,s in zip(front, side):
            front_features.append(self.feature_extractor(f.view(1, 1, 512, 512)).detach().numpy().reshape(256))
            side_features.append(self.feature_extractor(s.view(1, 1, 512, 512)).detach().numpy().reshape(256))
        
        front_features = np.array(front_features)
        side_features = np.array(side_features)

        return np.concatenate([front_features, side_features], axis = -1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='512_images.npy')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gender", type=str, default='male')
    parser.add_argument("--model", type=str, default='ae', help='pca or ae')
    parser.add_argument("--mode", type=str, default='features', help='features or performance')
    parser.add_argument("--loss", type=str, default='mse', help='choose one: mae, mse, mae+mse, bce')
    parser.add_argument("--dataset", type=str, default='calvis', help='calvis or nomo')

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.model == 'pca':
        mode = args.data_path.split("/")[-1].split('_')[0]
        data = np.load(args.data_path)
        data = np.array(data,dtype='float')/255.0
        data[data < 1] = 0
        data = 1 - data

        pca = PCAExtractor(gender = args.gender, loss = args.loss)
        
        if  args.mode == 'performance':
            
            print('PCA Evaluation Started ')
            pca.test_performance(data)
            print('Evaluation Done ')
        
        else:
            
            print('Extracting PCA Features .. ')
            features = pca.extract_features(data)
            
            if args.dataset == 'calvis':
                np.save(f"data/dataloaders/{args.gender}/{args.model}_{mode}_features.npy", features)
            else:
                np.save(f"nomodata/{args.gender}/{args.model}_{mode}_features.npy", features)
            print('Extraction Done ')

    if args.model == 'ae':
        model = AEXtractor(device=device, batch_size=args.batch_size, gender = args.gender, loss = args.loss)

        if args.mode == 'performance':

            dataset = Data(args.data_path)
            dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle=True, drop_last=True)
            print("AE Evaluation Started")
            model.test_performance(dataloader)
            print("Evaluation Done")
        
        else:
            mode = args.data_path.split("/")[-1].split('_')[0]
            data = np.load(args.data_path)
            data = np.array(data,dtype='float')/255.0
            data[data < 1] = 0
            data = 1 - data

            print("Extracting Encoder Features")
            features = model.extract_features(data)
            if args.dataset == 'calvis':
                np.save(f"data/dataloaders/{args.gender}/{args.model}_{mode}_features.npy", features)
            else:
                np.save(f"nomodata/{args.gender}/{args.model}_{mode}_features.npy", features)
                
            print("Extraction Done")
