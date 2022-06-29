import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from copy import deepcopy

class pd_dataset(Dataset):
    """
    Class to incorporate pandas DataFrame into PyTorch dataloader
    """
    def __init__(self, df, transform=None):
        """
        Args:
            df        : pandas DataFrame
            transform : Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        # Read the file and split the lines in a numpy array
        self.data = df.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Our sample is the element idx of the list self.data
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """
    Convert sample to Tensors.
    """
    def __call__(self, sample):
        x, y = sample
        return (torch.tensor([x]).float(),
                torch.tensor([y]).float())


class training_API(nn.Module):
    """
    Class implementing all training and evaluation tools.
    """
    def __init__(self, device):
        """
        Initializes required superseding methods.
        """
        super().__init__()
        super().train()
        super().zero_grad()

        self.device = device

    def train_model(self, train_loader, val_loader, num_epochs, loss_fn, optimizer, verbose=True):
        """
        Performs num_epochs of training on model
        Args:
            train_loader : Dataloader of training set
            val_loader   : Dataloader of validation set
            num_epochs   : Number of training epochs
            loss_fn      : Loss function
            optimizer    : Optimizer
        """
        self.history = dict()
        train_loss_log = np.zeros(num_epochs)
        val_loss_log = np.zeros(num_epochs)
        for epoch_num in range(num_epochs):
            train_loss = self.train_step(train_loader, loss_fn, optimizer)
            train_loss_log[epoch_num] = train_loss
            
            val_loss = self.evaluate(val_loader, loss_fn, verbose=False)
            if verbose:
                print(
                    f"Epoch: {epoch_num+1} >>> Training loss: {train_loss:.5f} | Validation loss: {val_loss:.5f}", end='\r')
            val_loss_log[epoch_num] = val_loss

        self.history['train'] = train_loss_log
        self.history['valid'] = val_loss_log
        self.history['epoch'] = np.array(range(1, num_epochs+1))
        return val_loss_log[-1]

    def train_step(self, dataloader, loss_fn, optimizer):
        """
        Performs one training epoch.
        Args:
            dataloader : Dataloader of training set
            loss_fn    : Loss function
            optimizer  : Optimizer
        """

        train_loss = []
        self.train()  # Training mode (e.g. enable dropout, batchnorm updates,...)
        for sample_batched in dataloader:
            # Move data to device
            x_batch = sample_batched[0].to(self.device)
            label_batch = sample_batched[1].to(self.device)

            #Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            out = self(x_batch)

            # Compute loss, back-propagate
            loss = loss_fn(out, label_batch)
            loss.backward()

            # Update the weights
            optimizer.step()
            # Save train loss for this batch
            loss_batch = loss.detach().cpu().numpy()
            train_loss.append(loss_batch)

        train_loss = np.mean(train_loss)
        return train_loss

    def evaluate(self, dataloader, loss_fn, verbose=True):
        self.eval()
        data_loss = []
        with torch.no_grad():
            for sample_batched in dataloader:
                x_batch = sample_batched[0].to(self.device)
                label_batch = sample_batched[1].to(self.device)

                out = self(x_batch)
                loss = loss_fn(out, label_batch)
                loss_batch = loss.detach().cpu().numpy()
                data_loss.append(loss_batch)

            data_loss = np.mean(data_loss)
            if verbose:
                print(f"Loss = {data_loss}")
            return data_loss

    def cross_validate(self, train_loader, test_loader, num_epochs, loss_fn, optimizer, k):
        """
        Cross-validate the model using k-fold.
        """
        composed_transform = transforms.Compose([ToTensor()])

        # Split the dataset in k folds
        samples_per_fold = train_loader.dataset.__len__() // k

        fold_losses = np.zeros((2, k, num_epochs))
        # Check the type of dataset used
        if train_loader.dataset.__class__ == pd_dataset:
            train_set_folds = [
                pd_dataset(
                    pd.DataFrame(
                    train_loader.dataset.data[
                        samples_per_fold*i:samples_per_fold*(i+1)]), transform=composed_transform)
                    for i in range(k)]

        else:
            train_set_folds = [train_loader.dataset[samples_per_fold*i:samples_per_fold*(i+1)] for i in range(k)]

        if self.__class__ == reg_model:
            model_fold = self.__class__(1, self.params, 1, self.device)
        # elif model.__class__ == CNN2d:
        #     pass

        # Create a copy of the optimizer to use on the fold model
        fold_optim = optimizer.__class__(
            model_fold.parameters(), lr=optimizer.param_groups[0]['lr'])

        for fold in range(k):
            print(f"Fold {fold+1}/{k}...", end='\t')
            # Re-initialize the weights of the model
            model_fold.reset_weights()
            # Create a deepcopy of the folds to avoid modifying the original list
            fold_copy = deepcopy(train_set_folds)
            # pops the k-th fold to use for validation
            valid_fold = DataLoader(fold_copy.pop(fold), batch_size = None)
            # makes training dataloader from the remaining folds
            train_fold = DataLoader(ConcatDataset(fold_copy),
                        batch_size=None, shuffle=True
                                )

            # Send the temporary model to the device
            model_fold.to(model_fold.device)

            # Train the model
            model_fold.train_model(
                train_fold, valid_fold, num_epochs, loss_fn, fold_optim, verbose=False)
            fold_losses[0,fold] = model_fold.history['train']
            fold_losses[1,fold] = model_fold.history['valid']
            print("Done.")

    
        self.load_state_dict(model_fold.state_dict())    
        self.history['train'], self.history['valid'] = np.mean(fold_losses, axis=1)
        self.history['epoch'] = np.array(range(1, num_epochs+1))
        
class reg_model(training_API):

    def __init__(self, N_input, params, N_out, device):
        """
        Args:
        n_input - Input size
        params - Dict of parameters for the model
        N_out - Output size
        device - Device to use

        The dictionary structure allows easier implementation of different models
        through hyperparameter tuning.
        """
        super().__init__(device)
        self.params = params
        self.act = getattr(nn, params['activation'])()

        self.model = nn.Sequential(
            nn.Linear(in_features=N_input, out_features=self.params['N_h1']),
            nn.Dropout(self.params['dropout']),
            self.act,
            nn.Linear(in_features=self.params['N_h1'], out_features=self.params['N_h2']),
            nn.Dropout(params['dropout']),
            self.act,
            nn.Linear(in_features=self.params['N_h2'], out_features=self.params['N_h3']),
            nn.Dropout(self.params['dropout']),
            self.act,
            nn.Linear(in_features=self.params['N_h3'], out_features=N_out)
        )

        #Initializes weights upon instantiation according to Glorot framework
        self.reset_weights()
        self.history = dict()
    def forward(self, x):
        return self.model(x)

    def reset_weights(self):
        self.model.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)