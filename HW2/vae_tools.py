import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import utils as vutils
import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
from faulthandler import disable

# Training function


class Encoder(nn.Module):

    def __init__(self, n, params):
        super().__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=params['conv1']['filters'],
                      kernel_size=params['conv1']['kernel'],
                      stride=params['conv1']['stride'],
                      padding=params['conv1']['padding']),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=params['conv1']['filters'], out_channels=params['conv2']['filters'],
                      kernel_size=params['conv2']['kernel'],
                      stride=params['conv2']['stride'],
                      padding=params['conv2']['padding']),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=params['conv2']['filters'], out_channels=params['conv3']['filters'],
                      kernel_size=params['conv3']['kernel'],
                      stride=params['conv3']['stride'],
                      padding=params['conv3']['padding']),
            nn.ReLU(inplace=True)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(in_features=n, out_features=params['lin1']),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=params['lin1'],
                      out_features=params['latent_space'])
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):

    # , conv1, conv2, conv3, lin1, n_side, encoded_space_dim):

    def __init__(self, conv_neurons, n_side, params):

        super().__init__()  # conv1, conv2, conv3, lin1, n_side, encoded_space_dim

        # Linear section

        self.decoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(in_features=params['latent_space'],
                      out_features=params['lin1']),
            nn.ReLU(inplace=True),
            # Second linear layer
            nn.Linear(in_features=params['lin1'],
                      out_features=conv_neurons),
            nn.ReLU(inplace=True)
        )
        # Unflatten
        self.unflatten = nn.Unflatten(dim=-1, unflattened_size=(
            params['conv3']['filters'], n_side, n_side)
        )
        # Convolutional section
        self.decoder_conv = nn.Sequential(
            # First transposed convolution
            nn.ConvTranspose2d(in_channels=params['conv3']['filters'], out_channels=params['conv2']['filters'],
                               kernel_size=params['conv2']['kernel'],
                               stride=params['conv2']['stride'],
                               output_padding=params['conv3']['padding']),
            nn.ReLU(True),
            # Second transposed convolution
            nn.ConvTranspose2d(in_channels=params['conv2']['filters'], out_channels=params['conv1']['filters'],
                               kernel_size=params['conv1']['kernel'],
                               stride=params['conv2']['stride'],
                               padding=params['conv2']['padding'],
                               output_padding=params['conv2']['padding']),
            nn.ReLU(True),
            # Third transposed convolution
            nn.ConvTranspose2d(in_channels=params['conv1']['filters'], out_channels=1,
                               kernel_size=params['conv1']['kernel'],
                               stride=params['conv1']['stride'],
                               padding=params['conv1']['padding'],
                               output_padding=params['conv1']['padding'])
        )

    def forward(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Unflatten
        x = self.unflatten(x)
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        # Apply a sigmoid to force the output to be between 0 and 1 (valid pixel values)
        x = torch.sigmoid(x)
        return x


class Autoencoder(nn.Module):

    def __init__(self, in_side, params, device, keep_loss = True):
        super().__init__()
        self.n_side1 = self.no_features(in_side, params['conv1'])
        self.n_side2 = self.no_features(self.n_side1, params['conv2'])
        self.n_side3 = self.no_features(self.n_side2, params['conv3'])
        self.lin_neurs = self.n_side3**2 * params['conv3']['filters']
        self.latent_space = params['latent_space']
        self.decoder = Decoder(self.lin_neurs, self.n_side3, params)
        self.encoder = Encoder(self.lin_neurs, params)
        self.epochs_trained = 0
        self.device = device
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.keep_loss = keep_loss
        if keep_loss:
            self.loss_history = {'training': [], 'validation': []}

    def no_features(self, n, conv):
        n_out = (n - conv['kernel'] + 2 * conv['padding'])//conv['stride'] + 1
        return n_out

    def forward(self, x):
        latent_space = self.encoder(x)
        x = self.decoder(latent_space)
        return x


def train_epoch(autoencoder: Autoencoder, device: torch.device, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.modules.loss, optimizer: torch.optim, verbose: bool) -> float:
    # Set train mode for both the encoder and the decoder
    loss_fn.to(device)
    autoencoder.train()
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    # with "_" we just ignore the labels (the second element of the dataloader tuple)
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
                f'\rpartial train loss (single batch): {loss.data:4f}', end='')
    if verbose:
        print(
            f'partial train loss (single batch): {loss.data:4f}', end='\tFinding validation loss... ')

    return loss.data


def test_epoch(autoencoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    autoencoder.to(device)
    loss_fn.to(device)
    autoencoder.eval()
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


def train_AE(autoencoder, train_dataloader, test_dataloader, num_epochs, loss_fn, optim, device, save_dir=None, verbose=True):
    if autoencoder.keep_loss:
        autoencoder.loss_history['training'] = [*autoencoder.loss_history['training'], *[0]*num_epochs]
        autoencoder.loss_history['validation'] = [*autoencoder.loss_history['validation'], *[0]*num_epochs]
    best_loss_val = 9999999.0
    best_epoch = 0
    epoch_zero = autoencoder.epochs_trained

    if save_dir != None:
        params_path = f'{save_dir}/params'
        os.makedirs(params_path, exist_ok=True)

        plots_path = f'{save_dir}/plots'
        os.makedirs(plots_path, exist_ok=True)

    for epoch in range(epoch_zero, epoch_zero + num_epochs):
        if verbose:
            print(
                f'EPOCH {epoch +1}/{epoch_zero + num_epochs} : ', end='\n')
        # Training (use the training function)
        train_loss = train_epoch(
            autoencoder,
            device=device,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optim,
            verbose=verbose)

        # Validation (use the testing function)
        val_loss = test_epoch(
            autoencoder,
            device=device,
            dataloader=test_dataloader,
            loss_fn=loss_fn)

        # Print Validation loss
        if verbose:
            print(f'Validation loss: {val_loss:4f}', end='\n')

        # Store losses in dict
        if autoencoder.keep_loss:
            autoencoder.loss_history['training'][epoch] = train_loss.item()
            autoencoder.loss_history['validation'][epoch] = val_loss.item()

        # Plot progress
        # Get the output of a specific image (the test image at index 0 in this case)
        img = train_dataloader.dataset.data[0].unsqueeze(0).to(device)
        autoencoder.eval()
        autoencoder.epochs_trained += 1
        if save_dir != None:
            if val_loss.item() < best_loss_val:
                best_epoch = epoch
                best_loss_val = val_loss
                torch.save(autoencoder.state_dict(),
                           f'{save_dir}/params/t{epoch + 1}.pth')
            plt.ioff()
            fig, axs = plot_inout(autoencoder, test_dataloader.dataset, device, idx = 39)
            fig.savefig(f'{plots_path}/t={epoch + 1}.jpg')
            plt.close()
    if verbose:
        print(f'Best loss = {best_loss_val:.4f} in epoch {best_epoch}')
    return best_loss_val, best_epoch



def plot_inout(autoencoder, dataset, device, idx = None):
    if idx == None:
        img, _ = dataset[random.randint(0, len(dataset))]
    else:
        img, _ = dataset[idx]
    img = img.unsqueeze(0) # Add the batch dimension in the first axis
    # Encode the image

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

class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(
                params['latent_space'], params['G_filters'] * 4, 3, 2, 0, bias=False),
            nn.BatchNorm2d(params['G_filters'] * 4),
            nn.ReLU(True),
            # state size. (G_filters*8, 4, 4)
            nn.ConvTranspose2d(
                params['G_filters'] * 4, params['G_filters'] * 2, 3, 2, 0, bias=False),
            nn.BatchNorm2d(params['G_filters'] * 2),
            nn.ReLU(True),
            # state size. (G_filters*4, 8, 8)
            nn.ConvTranspose2d(params['G_filters'] * 2,
                               params['G_filters'], 3, 2, 0, bias=False),
            nn.BatchNorm2d(params['G_filters']),
            nn.ReLU(True),
            # state size. (G_filters*2, 16, 16)
            nn.ConvTranspose2d(
                params['G_filters'], params['n_channels'], 3, 2, 2, 1, bias=False),
            nn.Tanh()
            # state size. (n_channels, 64, 64)
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, params, conditional=False):
        super(Discriminator, self).__init__()
        self.conditional = conditional
        self.main = nn.Sequential(   # input is (1,28,28)
                                  nn.Conv2d(in_channels=params['n_channels'], out_channels=params['D_filters'],
                                            kernel_size=4, stride=2, padding=1),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  # state size (D_filters, 14, 14)
                                  nn.Conv2d(in_channels=params['D_filters'], out_channels=params['D_filters'] * 2,
                                            kernel_size=4, stride=2, padding=1),
                                  nn.BatchNorm2d(params['D_filters'] * 2),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  # state size (D_filters*2, 7, 7)
                                  nn.Conv2d(in_channels=params['D_filters'] * 2, out_channels=params['D_filters'] * 4,
                                            kernel_size=4, stride=2, padding=1),
                                  nn.BatchNorm2d(params['D_filters'] * 4),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  # state size (D_filters*4, 3, 3)
                                  nn.Conv2d(in_channels=params['D_filters'] * 4, out_channels=1,
                                            kernel_size=4, stride=2, padding=1)
                                  # scalar output (1, 1, 1)
                                  )
    def forward(self, input):
        return self.main(input)

class GAN(nn.Module):
    def __init__(self, params, device, conditional=False):
        super(GAN, self).__init__()
        self.conditional = conditional
        self.latent_space = params['latent_space']
        self.netG = Generator(params)
        self.netD = Discriminator(params, conditional = self.conditional)
        self.epochs_trained = 0
        self.device = device
        
        self.optimizerD = getattr(torch.optim, params['opt'])([{'params': self.netD.parameters()},], lr=params['lr'])
        self.optimizerG = getattr(torch.optim, params['opt'])([{'params': self.netG.parameters()},], lr=params['lr'])

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
            device, saturating, snapshots = True, save_dir=None, verbose=True):
    
    if snapshots:
        img_list = []
        iters = 0

    loss_history = {'G_losses' : [0]*num_epochs, 'D_losses' : [0]*num_epochs}
    G_losses, D_losses = [], []
    fixed_noise = torch.randn(64, gan.latent_space, 1, 1, device=device)
    for epoch in range(gan.epochs_trained, gan.epochs_trained + num_epochs):
    # For each batch in the dataloader
        for i, data in enumerate(train_dataloader):#, desc=f'Training epoch #{epoch + 1}/{gan.epochs_trained + num_epochs}',disable=not(verbose)):


            output, label, fake, D_x, D_G_z1, errD = update_D(gan, data, criterion, device)

            D_G_z2, errG = update_G(gan, output, label, fake, criterion, saturating=saturating)
            
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if snapshots:
                if (iters % 500 == 0) or ((epoch == num_epochs) and (i == len(train_dataloader)-1)):
                    with torch.no_grad():
                        fake = gan.netG(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1
        if verbose:
            print(f'[{epoch + 1}/{gan.epochs_trained + num_epochs}]\tLoss_D: {errD.item():.4f}, \tLoss_G: {errG.item():.4f}, \tD(x): {D_x:.4f} \tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
        
        loss_history['G_losses'][epoch] = np.mean(G_losses)
        loss_history['D_losses'][epoch] = np.mean(D_losses)
    gan.epochs_trained += num_epochs
    if snapshots:
        return loss_history, img_list
    else:
        return loss_history
