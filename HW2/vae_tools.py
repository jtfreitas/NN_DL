import os
from copy import deepcopy
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import utils as vutils

import numpy as np
import matplotlib.pyplot as plt


################################################################

# Autoencoder modules

class Encoder(nn.Module):
    """
    Encoder for the AE.
    """
    def __init__(self, conv_out, params, init_weight=None):
        super().__init__()
        self.init_weight = init_weight
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=params['conv1']['filters'],
                kernel_size=params['conv1']['kernel'],
                stride=params['conv1']['stride'],
                padding=params['conv1']['padding']),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(params['conv1']['filters']),
            nn.Conv2d(
                in_channels=params['conv1']['filters'],
                out_channels=params['conv2']['filters'],
                kernel_size=params['conv2']['kernel'],
                stride=params['conv2']['stride'],
                padding=params['conv2']['padding']),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(params['conv2']['filters']),
            nn.Conv2d(
                in_channels=params['conv2']['filters'],
                out_channels=params['conv3']['filters'],
                kernel_size=params['conv3']['kernel'],
                stride=params['conv3']['stride'],
                padding=params['conv3']['padding']),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(params['conv3']['filters'])
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(
                in_features=np.prod(conv_out),
                out_features=params['lin1']),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=params['lin1'],
                out_features=params['latent_space'])
        )

    def forward(self, x):
        """
        Forward pass of the encoder.
        """
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

class Decoder(nn.Module):
    """
    Decoder for the AE.
    """
    def __init__(self, conv_in, params, init_weight=None):

        super().__init__()

        # Linear section
        self.init_weight = init_weight
        self.decoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(
                in_features=params['latent_space'],
                out_features=params['lin1']),
            nn.ReLU(inplace=True),
            #Second linear layer
            nn.Linear(
                in_features=params['lin1'],
                out_features=np.prod(conv_in)),
            nn.ReLU(inplace=True)
        )
        # Unflatten
        self.unflatten = nn.Unflatten(dim=-1, unflattened_size=conv_in)
        # Convolutional section
        self.decoder_conv = nn.Sequential(
            # First transposed convolution
            nn.BatchNorm2d(params['conv3']['filters']),
            nn.ConvTranspose2d(
                in_channels=params['conv3']['filters'],
                out_channels=params['conv2']['filters'],
                kernel_size=params['conv2']['kernel'],
                stride=params['conv2']['stride'],
                output_padding=params['conv3']['padding']),
            nn.ReLU(True),
            nn.BatchNorm2d(params['conv2']['filters']),
            # Second transposed convolution
            nn.ConvTranspose2d(
                in_channels=params['conv2']['filters'],
                out_channels=params['conv1']['filters'],
                kernel_size=params['conv1']['kernel'],
                stride=params['conv2']['stride'],
                padding=params['conv2']['padding'],
                output_padding=params['conv2']['padding']),
            nn.ReLU(True),
            nn.BatchNorm2d(params['conv1']['filters']),
            # Third transposed convolution
            nn.ConvTranspose2d(
                in_channels=params['conv1']['filters'],
                out_channels=1,
                kernel_size=params['conv1']['kernel'],
                stride=params['conv1']['stride'],
                padding=params['conv1']['padding'],
                output_padding=params['conv1']['padding'])
        )

    def forward(self, x):
        """
        Forward pass of the decoder.
        """
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

class Autoencoder(nn.Module):
    """
    Autoencoder main class.
    """
    def __init__(self, in_side, params, device, init_weight='normal'):
        super().__init__()
        self.init_weight = init_weight
        self.params = params
        self.conv_out = (
            self.params['conv3']['filters'],
            self.no_features(params, in_side),
            self.no_features(params, in_side)
            )
        self.latent_space = self.params['latent_space']
        self.decoder = Decoder(self.conv_out, self.params)
        self.encoder = Encoder(self.conv_out, self.params)
        self.device = device
        self.history = None

        self.encoder.to(self.device)
        self.decoder.to(self.device)

    def no_features(self, params, n):
        """
        Calculate the number of features in the convolutional layers.
        """
        for i in params.items():
            if 'conv' in i[0]:
                n = (n - i[1]['kernel'] + 2 * i[1]['padding'])//i[1]['stride'] + 1
        return n

    def forward(self, x):
        """
        Forward pass of the autoencoder.
        """
        latent_space = self.encoder(x)
        x = self.decoder(latent_space)
        return x

    def create_history(self, num_epochs):
        """
        Creates history dict to store losses during training.
        """
        self.history = dict(
            train=np.zeros(num_epochs),
            valid=np.zeros(num_epochs),
            epoch=np.arange(1,num_epochs+1)
            )

    def init_weights(self, m):
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

    def reset_weights(self):
        """
        Applies weight initialization on encoder and decoder
        """
        self.apply(self.init_weights)
        # self.decoder.apply(init_weights)


def train_epoch(autoencoder, device, dataloader, loss_fn, optimizer, verbose):
    """
    Performs one epoch of training.
    """
    # Set train mode for both the encoder and the decoder
    loss_fn.to(device)
    autoencoder.encoder.train()
    autoencoder.decoder.train()
    # Iterate over dataloader batches
    for image_batch, _ in dataloader:
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        decoded_data = autoencoder(image_batch)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        if verbose:
            print(
                f'\rpartial train loss (single batch): {loss.data:.4f}', end=' ')

    return loss.data


def test_epoch(autoencoder, device, dataloader, loss_fn):
    """
    Performs one epoch of testing.
    """
    # Set evaluation mode for encoder and decoder
    autoencoder.to(device)
    loss_fn.to(device)
    autoencoder.encoder.eval()
    autoencoder.decoder.eval()
    with torch.no_grad():  # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            # Encode-decode data
            decoded_data = autoencoder(image_batch)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data

# Training cycle

def train_AE(autoencoder, num_epochs, train_dataloader, loss_fn, optim, device,
         test_dataloader=None, save_dir=None, verbose=True):
    """
    Trains autoencoder model.
    """
    autoencoder.create_history(num_epochs)

    # Initialize fake loss values for comparison
    best_loss_train = 9999999.0
    if test_dataloader is not None:
        best_loss_val = 9999999.0
    best_epoch = 0
    # This helps avoid storing nans in the training timeline

    if save_dir is not None:
        params_path = f'{save_dir}/params'
        os.makedirs(params_path, exist_ok=True)

        plots_path = f'{save_dir}/plots'
        os.makedirs(plots_path, exist_ok=True)

    for epoch in range(num_epochs):
        if verbose:
            print(
                f'EPOCH {epoch+1}/{num_epochs} : ', end='\n')
        # Training (use the training function)
        train_loss = train_epoch(
            autoencoder,
            device=device,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optim,
            verbose=verbose)

        # Validation (use the testing function)
        if test_dataloader is not None:
            val_loss = test_epoch(
                autoencoder,
                device=device,
                dataloader=test_dataloader,
                loss_fn=loss_fn)

        # Print Validation loss
            if verbose:
                print(f'Validation loss: {val_loss:4f}', end='\n')

        # Store losses in dict
        autoencoder.history['train'][epoch] = train_loss.item()
        if test_dataloader is not None:
            autoencoder.history['valid'][epoch] = val_loss.item()

        if test_dataloader is not None:
            if val_loss.item() < best_loss_val:
                best_epoch = epoch
                best_loss_val = val_loss
                best_loss_train = train_loss
                best_state = autoencoder.state_dict()
        else:
            if train_loss.item() < best_loss_train:
                best_epoch = epoch
                best_loss_train = train_loss
                best_state = autoencoder.state_dict()

        if save_dir is not None:
            autoencoder.eval()
            #Save parameters for all epochs
            torch.save(autoencoder.state_dict(),
                       f'{save_dir}/params/t{epoch + 1}.pth')

            # Plot progress
            plt.ioff()
            fig, _ = plot_inout(autoencoder, train_dataloader.dataset, device, idx = 39)
            fig.savefig(f'{plots_path}/t={epoch}.jpg')
            plt.close()
    if verbose:
        if test_dataloader is not None:
            print(f'Best loss = {best_loss_val:.4f} in epoch {best_epoch}')
            print(f'Setting model state to best epoch: {best_epoch}')
        else:
            print(f'Best loss = {best_loss_train:.4f} in epoch {best_epoch}')
            print(f'Setting model state to best epoch: {best_epoch}')

    if test_dataloader is not None:
        autoencoder.load_state_dict(best_state)
        return best_loss_val, best_epoch
    else:
        autoencoder.load_state_dict(best_state)
        return best_loss_train, best_epoch

def CV_AE(k, autoencoder, num_epochs, train_loader, loss_fn, optim, verbose=True):
    """
    Validate the model using k-fold.
    This is only used to attest for low variance of the model
    in face of the data.
    """

    # Split the dataset in k folds
    samples_per_fold = len(train_loader.dataset) // k
    subset_idxs = np.array([range(samples_per_fold*i, samples_per_fold*(i+1))
        for i in range(k)])
    fold_losses = np.zeros((2, k, num_epochs))
    # Check the type of dataset used

    train_set_folds = [Subset(train_loader.dataset, idx_set) for idx_set in subset_idxs]

    model_fold = deepcopy(autoencoder)

    # Create a copy of the optimizer to use on the fold model
    fold_optim = optim.__class__(
        model_fold.parameters(), lr=optim.param_groups[0]['lr'])

    for fold in range(k):
        if verbose:
            print(f"Fold {fold+1}/{k}...", end='\t')
        # Re-initialize the weights of the model
        model_fold.reset_weights()
        # Create a deepcopy of the folds to avoid modifying the original list
        fold_copy = deepcopy(train_set_folds)
        # pops the k-th fold to use for validation
        valid_fold = DataLoader(fold_copy.pop(fold), batch_size = 1)
        # makes training dataloader from the remaining folds
        train_fold = DataLoader(ConcatDataset(fold_copy),
                    batch_size=train_loader.batch_size, shuffle=True
                            )
        # Send the temporary model to the device
        model_fold.to(model_fold.device)

        # Train the model
        train_AE(model_fold, num_epochs, train_fold,
            loss_fn, fold_optim, model_fold.device,
            test_dataloader = valid_fold, save_dir=None, verbose=False)

        fold_losses[0,fold] = model_fold.history['train']
        fold_losses[1,fold] = model_fold.history['valid']
        if verbose:
            print("Done.")

    return fold_losses

def plot_inout(autoencoder, dataset, device, idx = None):
    """
    Plot the input and output of the model.
    """
    if idx is None:
        img, _ = dataset[random.randint(0, len(dataset))]
    else:
        img, _ = dataset[idx]
    img = img.unsqueeze(0) # Add the batch dimension in the first axis

    fig, axs = plt.subplots(1,2, figsize=(14, 8), tight_layout='pad')

    img = img.to(device)
    autoencoder = autoencoder.to(device)
    img_dec = autoencoder(img)
    axs[0].imshow(np.reshape(img.cpu().detach().numpy(), (28,28)), cmap='gist_gray')
    axs[1].imshow(np.reshape(img_dec.cpu().detach().numpy(), (28,28)), cmap='gist_gray')

    axs[0].axis('off')
    axs[1].axis('off')
    axs[0].set_title('Original', fontsize=24)
    axs[1].set_title('Reconstructed', fontsize=24)
    return fig, axs

class Classifier(nn.Module):
    """
    Classifier for autoencoder
    """
    def __init__(self, params, device):
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(*params['lin1']),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(*params['lin2']),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(*params['lin3']),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(*params['lin4'])
        )
        self.history = None
        self.reset_weights()

    def forward(self, x):
        """
        Forward pass of the classifier.
        """
        return self.net(x)

    def predict(self, x):
        """
        Predict the class of the input.
        """
        return nn.Softmax(dim=1)(self.forward(x))

    def create_history(self, num_epochs):
        """
        Create a history of the loss for each epoch.
        """
        self.history = dict(
            train=np.zeros(num_epochs),
            valid=np.zeros(num_epochs),
            epoch=np.arange(1,num_epochs+1)
            )


    def reset_weights(self):
        """
        Reset weights of the classifier.
        """
        self.net.apply(self.init_weights)

    def init_weights(self, m):
        """
        Initialize weights of the classifier with Glorot initialization.
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

def train_model(model, train_loader, val_loader, num_epochs, loss_fn, optimizer,
                encoded = False, verbose=True):
    """
    Train the model.
        model: model to train
        train_loader: torch.utils.data.DataLoader
        val_loader: torch.utils.data.DataLoader
        num_epochs: int
        loss_fn: loss function
        optimizer: optimizer
        encoded: bool (if the data is encoded beforehand)
        verbose: bool
    """
    model.create_history(num_epochs)
    train_loss_log = np.zeros(num_epochs)
    val_loss_log = np.zeros(num_epochs)
    for epoch_num in range(num_epochs):
        train_loss = train_step(model, train_loader, loss_fn, optimizer, encoded=encoded)
        train_loss_log[epoch_num] = train_loss

        val_loss = evaluate(model, val_loader, loss_fn, encoded=encoded, verbose=verbose)
        if verbose:
            print(
                f"Epoch: {epoch_num+1} >>> Training loss: {train_loss:.5f}"
                "| Validation loss: {val_loss:.5f}", end='\r')
        val_loss_log[epoch_num] = val_loss

    model.history['train'] = train_loss_log
    model.history['valid'] = val_loss_log

    return val_loss_log[-1]

def train_step(model, dataloader, loss_fn, optimizer, encoded=False):
    """
    Train a single step of the model.
        model: model to train
        dataloader: torch.utils.data.DataLoader
        loss_fn: loss function
        optimizer: optimizer
        encoded: bool (if the data is encoded beforehand)
    """
    train_loss = []
    model.train()  # Training mode
    for sample_batched in dataloader:
        # Move data to device
        x_batch = sample_batched[0].to(model.device)
        label_batch = sample_batched[1].to(model.device)

        # Forward pass
        if encoded:
            out = model(x_batch)
        else:
            x_batch= x_batch.flatten(start_dim=1)
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

def evaluate(model, dataloader, loss_fn, encoded=False, verbose=True):
    """
    Evaluate the model.
        model: model to evaluate
        dataloader: torch.utils.data.DataLoader
        loss_fn: loss function
        encoded: bool (if the data is encoded beforehand)
        verbose: bool
    """
    model.eval()
    data_loss = []
    with torch.no_grad():
        for sample_batched in dataloader:
            x_batch = sample_batched[0].to(model.device)
            label_batch = sample_batched[1].to(model.device)
            if encoded:
                out = model(x_batch)
            else:
                x_batch = x_batch.flatten(start_dim=1)
                out = model(x_batch)
            loss = loss_fn(out, label_batch)
            loss_batch = loss.detach().cpu().numpy()
            data_loss.append(loss_batch)

        data_loss = np.mean(data_loss)
        if verbose:
            print(f"Loss = {data_loss:.5f}")
        return data_loss


#########################################################

# GAN modules


class Generator(nn.Module):
    """
    Generator component of the GAN
    """
    def __init__(self, params):
        """
        Initialize the generator.
        N.B: Output is square, so padding, stride and kernel
             are square too.
        """
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(
                params['latent_space'],
                params['G_filters'] * 4,
                3, 2, 0),
            nn.BatchNorm2d(params['G_filters'] * 4),
            nn.ReLU(True), # shape (G_filters*4, 3, 3)
            nn.ConvTranspose2d(
                params['G_filters'] * 4,
                params['G_filters'] * 2,
                3, 2, 0),
            nn.BatchNorm2d(params['G_filters'] * 2),
            nn.ReLU(True), # shape (G_filters*2, 7, 7)
            nn.ConvTranspose2d(
                params['G_filters'] * 2,
                params['G_filters'],
                3, 2, 0),
            nn.BatchNorm2d(params['G_filters']),
            nn.ReLU(True), # shape (G_filters, 15, 15)
            nn.ConvTranspose2d(
                params['G_filters'],
                params['n_channels'],
                3, 2, 2, 1, bias=False),
            nn.Tanh()      # shape (n_channels, 28, 28)
        )

    def forward(self, x):
        """
        Forward pass of the generator.
        """
        return self.main(x)


class Discriminator(nn.Module):
    """
    Discriminator component of the GAN
    """
    def __init__(self, params):
        super().__init__()
        self.main = nn.Sequential(   # input (1,28,28)
            nn.Conv2d(
                in_channels=params['n_channels'],
                out_channels=params['D_filters'],
                kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # shape (D_filters, 14, 14)
            nn.Conv2d(
                in_channels=params['D_filters'],
                out_channels=params['D_filters'] * 2,
                kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(params['D_filters'] * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # shape (D_filters*2, 7, 7)
            nn.Conv2d(
                in_channels=params['D_filters'] * 2,
                out_channels=params['D_filters'] * 4,
                kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(params['D_filters'] * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # shape (D_filters*4, 3, 3)
            nn.Conv2d(
                in_channels=params['D_filters'] * 4,
                out_channels=1,
                kernel_size=4, stride=2, padding=1)
            # scalar output (1, 1, 1)
            )
    def forward(self, x):
        """
        Forward pass of the discriminator.
        """
        return self.main(x)

class GAN(nn.Module):
    """
    GAN model
    NB: As opposed to the autoencoder, the GAN does not
        inherit from nn.Module, since there is no need to
        define a forward pass.
    """
    def __init__(self, params, device):
        """
        Initialize the model.
        params: dictionary of parameters
        device: torch.device
        """
        super().__init__()
        self.latent_space = params['latent_space']
        self.netG = Generator(params)
        self.netD = Discriminator(params) 
        self.epochs_trained = 0
        self.device = device

        self.optimizerD = getattr(torch.optim, params['opt'])(
            [{'params': self.netD.parameters()},], lr=params['lr'])
        self.optimizerG = getattr(torch.optim, params['opt'])(
            [{'params': self.netG.parameters()},], lr=params['lr'])
        self.reset_weights()
        

        self.to(self.device)  #Immediately send the model to the device 

    def reset_weights(self):
        """
        Reset the weights of the networks.
        """
        self.netD.apply(self.weights_init)
        self.netG.apply(self.weights_init)

    def weights_init(self, m):
        """
        Initialize the weights of the networks.
        """
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

def update_D(gan, data, criterion, device):
    """
    Maximize log(D(x)) + log(1 - D(G(z)))
    """
    real_label, fake_label = 1., 0.
    gan.netD.zero_grad()

    # Format batch
    real_cpu = data[0].to(device)
    b_size = real_cpu.size(0)
    label = torch.full((b_size,), real_label,
                    dtype=torch.float, device=device)

    # Forward pass real batch through D
    output = gan.netD(real_cpu).view(-1)

    # Compute loss
    errD_real = criterion(output, label)

    # Propagate loss
    errD_real.backward()
    D_x = output.mean().item()

    # Train with batch of latent vectors to generate fakes
    noise = torch.randn(b_size, gan.latent_space, 1, 1, device=device)
    fake = gan.netG(noise)
    label.fill_(fake_label)

    # Discriminate through fake images
    output = gan.netD(fake.detach()).view(-1)

    #Compute loss on fake batch, propagate loss
    errD_fake = criterion(output, label)
    errD_fake.backward()
    D_G_z1 = output.mean().item()

    #Error is sum of losses on real and fake batches
    errD = errD_real + errD_fake
    gan.optimizerD.step()

    return output, label, fake, D_x, D_G_z1, errD

def update_G(gan, output, label, fake, criterion, saturating=False):

    """
    Maximize log(D(G(z)))
    saturating: Whether to minmax log(1 - D(G(z)))) or maximize -log(D(G(z)))
    """
    real_label, fake_label = 1., 0.

    #set saturating or non-saturating loss
    label.fill_(fake_label) if saturating else label.fill_(real_label)
    output = gan.netD(fake).view(-1)
    errG = -criterion(output, label) if saturating else criterion(output, label)

    gan.netG.zero_grad()


    label.fill_(real_label)     # need real labels for G(z)
    #Compute loss, propagate
    output = gan.netD(fake).view(-1)
    errG = criterion(output, label)
    errG.backward()

    D_G_z2 = output.mean().item()

    #Update optimizer
    gan.optimizerG.step()

    return D_G_z2, errG


def train_GAN(gan, train_dataloader,num_epochs,
            criterion,
            device, saturating, snapshots = True, verbose=True):
    """
    Train the GAN model.
        gan: GAN model instance
        train_dataloader: torch.utils.data.DataLoader
        num_epochs: int
        criterion: loss function
        device: torch.device
        saturating: bool (MM or NS loss method)
        snapshots: bool
        verbose: bool
    """
    if snapshots:
        img_list = []
        iters = 0
    G_loss, D_loss = np.zeros(num_epochs), np.zeros(num_epochs)
    loss_history = {'G_losses' : G_loss, 'D_losses' : D_loss}

    fixed_noise = torch.randn(64, gan.latent_space, 1, 1, device=device)
    for epoch in range(num_epochs):
    # For each batch in the dataloader
        G_losses, D_losses = np.zeros(len(train_dataloader)), np.zeros(len(train_dataloader))
        for i, data in enumerate(train_dataloader):


            output, label, fake, D_x, D_G_z1, errD = update_D(
                gan, data, criterion, device)

            D_G_z2, errG = update_G(
                gan, output, label, fake, criterion, saturating=saturating)
            # Save Losses for plotting later
            G_losses[i] = errG.item()
            D_losses[i] = errD.item()

            # Visualize generator's performance based on fixed_noise
            if snapshots:
                if (iters % 500 == 0) or \
                    ((epoch == num_epochs) and (i == len(train_dataloader)-1)):
                    with torch.no_grad():
                        fake = gan.netG(fixed_noise).detach().cpu()
                    img_list.append(
                        vutils.make_grid(fake, padding=2, normalize=True)
                        )

                iters += 1
        if verbose:
            print(f'\r[{epoch + 1}/{num_epochs}]\t'
            f'Loss_D: {errD.item():.4f}, \tLoss_G: {errG.item():.4f}, \tD(x):'
            f'{D_x:.4f} \tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}', end=' ')

        D_loss[epoch] = np.mean(D_losses)
        G_loss[epoch] = np.mean(G_losses)

    if snapshots:
        return loss_history, img_list

    return loss_history

def test_netD(gan, test_loader, criterion):
    """
    Test the discriminator.
        gan: GAN
        test_loader: torch.utils.data.DataLoader
        criterion: loss function
    """
    gan.netD.eval() #set to evaluation mode
    loss_list = torch.zeros(
        len(test_loader),dtype=torch.float, device = gan.device
        )
    with torch.no_grad():
        for i, sample_batched in enumerate(test_loader):

            x_batch = sample_batched[0].to(gan.device)
            prediction = gan.netD(x_batch)
            ground_truth = torch.ones(  #Use ones as truth labels
                prediction.size(),dtype=torch.float, device = gan.device
                )
            loss_list[i] = criterion(prediction, ground_truth)

        test_loss = torch.mean(loss_list)
    return test_loss.item()