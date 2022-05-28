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
from joblib import dump

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


class PCATrainer(object):
    def __init__(self, num_components, gender):
        self.pca = PCA(n_components = num_components)
        self.gender = gender
    def fit_model(self,data):
        front_data, side_data = data[:,:,:,0], data[:,:,:,1]
        side_data = side_data.reshape(data.shape[0], data.shape[1]*data.shape[2])
        front_data = front_data.reshape(data.shape[0], data.shape[1]*data.shape[2])
        front_model_weights = self.pca.fit(front_data)
        side_model_weights = self.pca.fit(side_data)

        try:
            os.mkdir('weights')
        except:
            pass
        
        dump(front_model_weights, f'weights/pca_{self.gender}_front.joblib')
        dump(side_model_weights, f'weights/pca_{self.gender}_side.joblib')

class AETrainer(object):
    def __init__(self, device, gpu, gender, batch_size, loss):
        self.device = device
        self.gender = gender
        self.gpu = gpu
        self.batch_size = batch_size

        losses = {'mse' : nn.MSELoss(),\
                  'mae' : nn.L1Loss(),\
                  'mae+mse': MAASE(),\
                  'bce': nn.BCELoss()}
        
        self.loss = losses[loss]
        self.epochs = None
        self.accuracy = Accuracy()

    def train_model(self, data_loader, epochs = 50, learning_rate= 1e-4, betas = (0.9, 0.99)):

        # image_size is the size of input mask
        self.epochs = epochs

        feature_extractor = Deep2DEncoder(image_size= 512 , kernel_size=3, n_filters=32)
        decoder = Deep2DDecoder(image_size=512, kernel_size=3, n_filters=32)

        feature_extractor.to(self.device)
        decoder.to(self.device)

        optimizer_extractor = torch.optim.Adam(feature_extractor.parameters(), lr = learning_rate, betas= betas)
        optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr = learning_rate, betas= betas)

        reconstruction_criterion = self.loss

        cuda = True if torch.cuda.is_available() else False
        
        if cuda:
            reconstruction_criterion.cuda()
        
        start_time = time.time()

        for epoch in range(epochs):
            epoch_loss_front= 0
            epoch_loss_side = 0
            total_accuracy_front = 0
            total_accuracy_side = 0
            
            feature_extractor.train()
            decoder.train()
            feature_extractor.requires_grad_(True)
            decoder.requires_grad_(True)
            
            step=0

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
                    
                optimizer_extractor.zero_grad()
                optimizer_decoder.zero_grad()
                recon_loss.backward() 
                optimizer_extractor.step()
                optimizer_decoder.step()
                
                step+=1
            
            epoch_loss_front = epoch_loss_front / step
            epoch_loss_side = epoch_loss_side / step
            total_accuracy_front = total_accuracy_front / step
            total_accuracy_side = total_accuracy_side / step

            elapsed_time = time.time() - start_time 
            print('Epoch {}/{} Time={:.2f}s train_loss_front={:.4f} train_loss_side={:.4f} accuracy_front= {:.4f} accuracy_side= {:.4f}'.format(
                                    epoch +1, epochs,
                                    elapsed_time,
                                    epoch_loss_front, epoch_loss_side, total_accuracy_front, total_accuracy_side))

        
        try:
            os.mkdir("weights")
        except:
            pass

        torch.save(feature_extractor.state_dict(), f'weights/feature_extractor_{self.gender}_{epochs}.pth')
        torch.save(decoder.state_dict(), f'weights/decoder_{self.gender}_{epochs}.pth')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='1', help="GPU number")
    parser.add_argument("--data_path", type=str, default='512_images.npy')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gender", type=str, default='male')
    parser.add_argument("--model", type=str, default='ae', help='pca or ae')
    parser.add_argument("--loss", type=str, default='mse', help='choose one: mae, mse, mae+mse, bce')

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model == 'pca':
        data = np.load(args.data_path)
        data = np.array(data,dtype='float')/255.0
        data[data < 1] = 0
        data = 1 - data
        print('PCA Fit Started ')
        pca = PCATrainer(num_components = 256, gender = args.gender)
        pca.fit_model(data)
        print('Fit Done ')

    if args.model == 'ae':
        dataset = Data(args.data_path)
        dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle=True, drop_last=True)
        model = AETrainer(device=device, gpu=args.gpu, batch_size=args.batch_size, gender = args.gender, loss = args.loss)
        print("Training AE Started")
        model.train_model(dataloader, epochs=50)
        print("Training Done")
