### Just in case code

class CNN2D(data_API):
    
    def __init__(self, device, in_side, conv1_filters, k_size1, conv2_filters, k_size2, Nd1, Nd2, N_labels):
        """
        device - device used for computing
        in_side - side length of the square images to classify
        conv1_filters - no of filters in the first convolutional layer
        k_size1 - kernel size of each filter in conv1
        conv2_filters - no of filters in the second convolutional layer
        k_size2 - kernel size of each filter in conv2
        Nh1 - Neurons in the 2nd dense layer (first layer 
        Nh2 - Neurons in the 2nd hidden layer
        No - Output size
        """
        super().__init__(device)
        super().train_model

        self.conv_shape = self.no_params_conv(
                self.no_params_conv(
                    self.no_params_conv(self.no_params_conv(in_side, k_size1, 0, 1),2, 0, 1),
                    2, 0, 1),
                k_size2, 0, 1)
        self.dropout = nn.Dropout(p = 0.4)
        
        self.model = nn.Sequential(
            nn.Conv2d(1, conv1_filters, k_size1), nn.ReLU(),
            nn.MaxPool2d(2, stride = 1),
            nn.Conv2d(conv1_filters, conv2_filters, k_size2), nn.ReLU(),
            nn.MaxPool2d(2, stride = 1),
            nn.Flatten(),
            nn.Linear(in_features=conv2_filters*self.conv_shape**2, out_features=Nd1), nn.ReLU(),
            nn.Linear(in_features=Nd1, out_features=Nd2), nn.ReLU(),
            nn.Linear(in_features=Nd2, out_features=N_labels)
        )

        print("Network initialized")
        
    def forward(self, x, additional_out=False):
        return self.model(x)

    def no_params_conv(self, in_size, kernel, padding, stride):
        """
        Finds no of parameters per channel after every convolution/pooling
        """
        return int((in_size - kernel + 2*padding)/stride + 1)

    def accuracy(self, dataset):

        dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers = 0)
        n_samples = dataloader.__len__()
        n_correct = 0

        self.eval()
        with torch.no_grad():
            for data in dataloader:
                x_sample = data[0].to(self.device)
                label = data[1].to(self.device)
                out = self(x_sample)
                out = F.softmax(out, dim = 1)
                if (((out[0] == out.max()).nonzero() == label).squeeze()).item() == True:
                    n_correct += 1
        return n_correct/n_samples