import torch
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class CsvDataset(Dataset):

    def __init__(self, df, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        # Read the file and split the lines in a list
        self.data = df.values
        
        # Each element of the list self.data is a tuple: (input, output)

    def __len__(self):
        # The length of the dataset is simply the length of the self.data list
        return len(self.data)

    def __getitem__(self, idx):
        # Our sample is the element idx of the list self.data
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
    
class ToTensor(object):
    """Convert sample to Tensors."""

    def __call__(self, sample):
        x, y = sample
        return (torch.tensor([x]).float(),
                torch.tensor([y]).float())

    
class reg_model(nn.Module):
    
    def __init__(self, N_input, N_h1, N_h2, N_h3, N_out):
        """
        n_input - Input size
        N_h1 - Neurons in the 1st hidden layer
        N_h2 - Neurons in the 2nd hidden layer
        N_out - Output size
        """
        super().__init__()
        super().train() 
        super().zero_grad()
        print('Network initialized')
        
        self.fc1 = nn.Linear(in_features = N_input, out_features = N_h1)
        self.fc2 = nn.Linear(in_features = N_h1, out_features = N_h2)
        self.fc3 = nn.Linear(in_features = N_h2, out_features = N_h3)
        self.out = nn.Linear(in_features = N_h3, out_features = N_out)
        self.act = nn.Sigmoid()

    def forward(self, x, additional_out=False):
        
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.out(x)
        return x
    
    def train_model(self, train_loader, val_loader, num_epochs, loss_fn, optimizer, verbose = True):
        train_loss_log = []
        val_loss_log = []
        for epoch_num in range(num_epochs):

            ### TRAIN
            train_loss= []
            self.train() # Training mode (e.g. enable dropout, batchnorm updates,...)
            for sample_batched in train_dataloader:
                # Move data to device
                x_batch = sample_batched[0].to(device)
                label_batch = sample_batched[1].to(device)

                # Forward pass
                out = self(x_batch)

                # Compute loss
                loss = loss_fn(out, label_batch)

                # Backpropagation
                self.zero_grad()
                loss.backward()

                # Update the weights
                optimizer.step()

                # Save train loss for this batch
                loss_batch = loss.detach().cpu().numpy()
                train_loss.append(loss_batch)

            train_loss = np.mean(train_loss)
            train_loss_log.append(train_loss)

            ### VALIDATION
            val_loss= []
            regression.eval() # Evaluation mode (e.g. disable dropout, batchnorm,...)
            with torch.no_grad(): # Disable gradient tracking
                for sample_batched in val_dataloader:
                    # Move data to device
                    x_batch = sample_batched[0].to(device)
                    label_batch = sample_batched[1].to(device)

                    # Forward pass
                    out = self(x_batch)

                    # Compute loss
                    loss = loss_fn(out, label_batch)

                    # Save val loss for this batch
                    loss_batch = loss.detach().cpu().numpy()
                    val_loss.append(loss_batch)

                # Save average validation loss
                val_loss = np.mean(val_loss)
                if verbose:
                    print(f"Epoch: {epoch_num} :::::::::: AVERAGE VAL LOSS: {np.mean(val_loss):.5f}", end = '\r')
                val_loss_log.append(val_loss)
                
        self.train_history = train_loss_log
        self.val_history = val_loss_log

class data_API(nn.Module):
    
    def __init__(self, device):
        super().__init__()
        super().train()
        super().zero_grad()
        
        self.device = device
    def train_model(self, train_loader, num_epochs, loss_fn, optimizer, val_loader = None, verbose = True):
        
        self.loss = loss_fn
        if val_loader != None:
            self.val_history = []
        train_loss_log = []
        val_loss_log = []
        for epoch_num in range(num_epochs):
            ### TRAIN
            train_loss= []
            self.train() # Training mode (e.g. enable dropout, batchnorm updates,...)
            for sample_batched in train_loader:
                # Move data to device
                x_batch = sample_batched[0].to(self.device)
                label_batch = sample_batched[1].to(self.device)
                # Forward pass
                out = self(x_batch)
                # Compute loss
                loss = loss_fn(out, label_batch)
                # Backpropagation
                self.zero_grad()
                loss.backward()
                # Update the weights
                optimizer.step()
                # Save train loss for this batch
                loss_batch = loss.detach().cpu().numpy()
                train_loss.append(loss_batch)
            train_loss = np.mean(train_loss)
            train_loss_log.append(train_loss)
            ### VALIDATION
            
            #######################
            if val_loader != None:
                self.val_history.append(self.write_validation(val_loader, verbose, epoch_num))
                
        self.train_history = train_loss_log
    
    def write_validation(self, val_loader, verbose, epoch_num):
        val_loss= []
        self.eval() # Evaluation mode (e.g. disable dropout, batchnorm,...)
        with torch.no_grad(): # Disable gradient tracking
            for sample_batched in val_loader:
                # Move data to device
                x_batch = sample_batched[0].to(self.device)
                label_batch = sample_batched[1].to(self.device)
                # Forward pass
                out = self(x_batch)
                # Compute loss
                the_loss = self.loss(out, label_batch)
                # Save val loss for this batch
                loss_batch = the_loss.detach().cpu().numpy()
                val_loss.append(loss_batch)
            # Save average validation loss
            val_loss = np.mean(val_loss)
            if verbose:
                print(f"Epoch: {epoch_num} :::::::::: AVERAGE VAL LOSS: {np.mean(val_loss):.5f}", end = '\r')
        return val_loss
        
    def test_model(self, test_loader, loss_fn):
        with torch.no_grad():
            x = sample[0].to(self.device)
            label = sample[1].to(self.device)
            # Forward pass
            out = self(x)
            # Compute loss
            loss = loss_fn(out, label)
            # Save test loss for this batch
            test_loss = loss.detach().cpu().numpy()
        self.out = out
        self.test_loss = test_loss
        
    def class_accuracy(self, val_loader):
        val_loss= []
        self.eval() # Evaluation mode (e.g. disable dropout, batchnorm,...)
        with torch.no_grad(): # Disable gradient tracking

            # Move data to device
            x_batch = sample_batched[0].to(self.device)
            label_batch = sample_batched[1].to(self.device)
            # Forward pass
            val_dataloader.dataset.targets
            out = self(x_batch)
                