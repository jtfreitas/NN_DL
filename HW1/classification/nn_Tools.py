import torch
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np


class pd_dataset(Dataset):

    def __init__(self, df, transform=None):
        """
        Args:
            df : pandas dataframe
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

def train_model(model, train_loader, val_loader, num_epochs, loss_fn, optimizer, verbose=True):
    model.create_history(num_epochs)
    train_loss_log = np.zeros(num_epochs)
    val_loss_log = np.zeros(num_epochs)
    for epoch_num in range(num_epochs):
        train_loss = train_step(model, train_loader, loss_fn, optimizer)
        train_loss_log[epoch_num] = train_loss
        
        val_loss = evaluate(model, val_loader, loss_fn, verbose=False)
        if verbose:
            print(
                f"Epoch: {epoch_num+1} >>> Training loss: {train_loss:.5f} | Validation loss: {val_loss:.5f}", end='\r')
        val_loss_log[epoch_num] = val_loss
    return val_loss_log[-1]

def train_step(model, dataloader, loss_fn, optimizer):
    train_loss = []
    model.train()  # Training mode (e.g. enable dropout, batchnorm updates,...)
    for sample_batched in dataloader:
        # Move data to device
        x_batch = sample_batched[0].to(model.device)
        label_batch = sample_batched[1].to(model.device)

        # Forward pass
        out = model(x_batch)

        # Compute loss
        loss = loss_fn(out, label_batch)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Update the weights
        optimizer.step()

        # Save train loss for this batch
        loss_batch = loss.detach().cpu().numpy()
        train_loss.append(loss_batch)

    train_loss = np.mean(train_loss)
    return train_loss

def evaluate(model, dataloader, loss_fn, verbose=True):
    model.eval()
    data_loss = []
    with torch.no_grad():
        for sample_batched in dataloader:
            x_batch = sample_batched[0].to(model.device)
            label_batch = sample_batched[1].to(model.device)

            out = model(x_batch)
            loss = loss_fn(out, label_batch)
            loss_batch = loss.detach().cpu().numpy()
            data_loss.append(loss_batch)

        data_loss = np.mean(data_loss)
        if verbose:
            print(f"Loss = {data_loss}")
        return data_loss

def write_validation(model, val_loader):
    val_loss = []
    model.eval()  # Evaluation mode (e.g. disable dropout, batchnorm,...)
    with torch.no_grad():  # Disable gradient tracking
        for sample_batched in val_loader:
            # Move data to device
            x_batch = sample_batched[0].to(model.device)
            label_batch = sample_batched[1].to(model.device)
            # Forward pass
            out = model(x_batch)
            # Compute loss
            the_loss = model.loss(out, label_batch)
            # Save val loss for this batch
            loss_batch = the_loss.detach().cpu().numpy()
            val_loss.append(loss_batch)
        # Save average validation loss
        val_loss = np.mean(val_loss)
    return val_loss

class reg_model(nn.Module):

    def __init__(self, N_input, N_h1, N_h2, N_h3, N_out, device):
        """
        n_input - Input size
        N_h1 - Neurons in the 1st hidden layer
        N_h2 - Neurons in the 2nd hidden layer
        N_out - Output size
        """
        super().__init__(device)

        print('Network initialized')

        self.fc1 = nn.Linear(in_features=N_input, out_features=N_h1)
        self.fc2 = nn.Linear(in_features=N_h1, out_features=N_h2)
        self.fc3 = nn.Linear(in_features=N_h2, out_features=N_h3)
        self.out = nn.Linear(in_features=N_h3, out_features=N_out)
        self.act = nn.Sigmoid()

    def forward(self, x):

        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.out(x)
        return x

    def create_history(self, num_epochs):
        self.history = dict(
            train=np.zeros(num_epochs),
            valid=np.zeros(num_epochs),
            epoch=np.arange(1, num_epochs+1))

class CNN2d(nn.Module):

    def __init__(self, in_side, params, device, init_weight = 'uniform', batch_norm=False):
        super().__init__()
        super().train()
        super().zero_grad()

        self.device = device
        self.params = params
        self.weights_after_conv1    = self.no_params_conv(in_side, params['conv1']['k1'], 0, 1)
        self.weights_after_maxpool1 = self.no_params_conv(self.weights_after_conv1,2, 1, 1)
        self.weights_after_conv2    = self.no_params_conv(self.weights_after_maxpool1, params['conv2']['k2'], 0, 1)
        self.weights_after_maxpool2 = self.no_params_conv(self.weights_after_conv2, 2, 0, 1)
        self.weights_after_conv3    = self.no_params_conv(self.weights_after_maxpool2, params['conv3']['k3'], 0, 1)
        self.weights_after_maxpool3 = self.no_params_conv(self.weights_after_conv3, 2, 0, 1)
        self.init_weight = init_weight
        
        #Convolutional component
        if batch_norm:
            self.features = nn.Sequential(
                #First layer
                nn.Conv2d(in_channels = 1,
                        out_channels = self.params['conv1']['filters1'],
                        kernel_size = self.params['conv1']['k1'],
                        padding='valid'),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride = 1, padding=1),
                nn.BatchNorm2d(self.params['conv1']['filters1']),
                #Second layer
                nn.Conv2d(in_channels = self.params['conv1']['filters1'],
                        out_channels = self.params['conv2']['filters2'],
                        kernel_size = self.params['conv2']['k2'],
                        padding='valid'),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride = 1),
                nn.BatchNorm2d(self.params['conv2']['filters2']),
                #Third layer
                nn.Conv2d(in_channels = self.params['conv2']['filters2'],
                        out_channels = self.params['conv3']['filters3'],
                        kernel_size = self.params['conv3']['k3'],
                        padding='valid'),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride = 1),
                nn.BatchNorm2d(self.params['conv3']['filters3'])
            )
        else:
            self.features = nn.Sequential(
                #First layer
                nn.Conv2d(in_channels = 1,
                        out_channels = self.params['conv1']['filters1'],
                        kernel_size = self.params['conv1']['k1'],
                        padding='valid'),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride = 1, padding=1),
                #Second layer
                nn.Conv2d(in_channels = self.params['conv1']['filters1'],
                        out_channels = self.params['conv2']['filters2'],
                        kernel_size = self.params['conv2']['k2'],
                        padding='valid'),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride = 1),
                #Third layer
                nn.Conv2d(in_channels = self.params['conv2']['filters2'],
                        out_channels = self.params['conv3']['filters3'],
                        kernel_size = self.params['conv3']['k3'],
                        padding='valid'),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride = 1),
            )
        #Dense NN component
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.params['conv3']['filters3']*self.weights_after_maxpool3**2,
                      out_features=self.params['lin1']),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=self.params['lin1'],
                      out_features=self.params['lin2']),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.2),
            nn.Linear(in_features=self.params['lin2'],
                      out_features=10)
        )
        #Weight initialization
        self.reset_model()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def no_params_conv(self, in_size, kernel, padding, stride):
        """
        Finds no of parameters per channel after every convolution/pooling.
        """
        return int((in_size - kernel + 2*padding)/stride + 1)

    def weight_init(self, m):
        """
        Initializes model weights according to Kaiming
        normal or uniform schemes.
        """
        if self.init_weight == 'normal':
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.01)
        elif self.init_weight == 'uniform':
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                m.bias.data.fill_(0.01)
                
    def predict(self, x):
        """
        Performs forward pass and softmax activation
        to perform class prediction.
        """
        x = self.forward(x)
        return nn.Softmax(1)(x)

    def reset_model(self):
        """
        Applies weight initialization on the model on-call.
        """
        self.features.apply(self.weight_init)
        self.classifier.apply(self.weight_init)
        

    def score(self, dataloader):
        """
        Calculates model score as a percentage accuracy.
        Takes the most probable class as the
        model prediction for each data point.
        """
        correct_preds = 0
        total_pts = 0
        dataloader_pts = dataloader.__len__()
        with torch.no_grad():
            for i, sample_batched in enumerate(dataloader):
                x_batch = sample_batched[0].to(self.device)
                out = self.predict(x_batch)
                pred_labels = np.argmax(out.cpu().detach().numpy(), axis=1)
                total_pts += len(pred_labels)
                correct_preds += np.sum(pred_labels == sample_batched[1].numpy())
                print(f"[{i}/{dataloader_pts}] >>> Accuracy score: {correct_preds/total_pts:.5f}", end='\r')
        return correct_preds/total_pts

    def create_history(self, num_epochs):
        self.history = dict(
            train=np.zeros(num_epochs),
            valid=np.zeros(num_epochs),
            epoch=np.arange(1,num_epochs+1))
