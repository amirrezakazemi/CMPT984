import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Auto Encoder class
class AE(nn.Module):
    def __init__(self, input_dim, encoder_dims, z_dim, decoder_dims):
        super(AE, self).__init__()
        # input_dim = is the dimension of your input
        # encoder_dims = is list containing of some integer that shows the dimension of encoder layers, between input layer and latent layer
        # z_dim = dimension of latent layer
        # decoder_dims = is list containing of some integer that shows the dimension of decoder layers, between latent layer and output layer (same as input_dim)

        self.type_str = 'AE'
        self.z_dim = z_dim  # latent space dimension
        self.dropout = nn.Dropout(p=0.2)  # dropout layer
        self.encoder, self.z_layer, self.decoder = None, None, None

        ### Stacking encoder layers
        encoder_layers = list()
        all_enc_dims = encoder_dims
        all_enc_dims.insert(0, input_dim)
        for i in range(len(all_enc_dims) - 1):
            encoder_layers.append(nn.Linear(all_enc_dims[i], all_enc_dims[i + 1]))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Compute Z
        self.z_layer = nn.Linear(all_enc_dims[-1], self.z_dim)

        # Stacking decoder layers
        decoder_layers = list()
        all_dec_dims = decoder_dims
        all_dec_dims.insert(0, self.z_dim)
        for i in range(len(all_dec_dims) - 1):
            decoder_layers.append(nn.Linear(all_dec_dims[i], all_dec_dims[i + 1]))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(all_dec_dims[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x_hat, z = None, None
        # x_hat is reconstructed output, z is latent representation
        dropped_out = self.dropout(x)
        temp = self.encoder(dropped_out)
        z = self.z_layer(temp)
        x_hat = self.decoder(z)
        return x_hat, z, None

    def get_loss(self, x, x_hat, *_):
        # Mean Squared Error
        return ((x_hat - x) ** 2).mean()


# Variational Auto Encoder class
class VAE(AE):
    def __init__(self, input_dim, encoder_dims, z_dim, decoder_dims):
        super(VAE, self).__init__(input_dim, encoder_dims, z_dim, decoder_dims)

        self.type_str = 'VAE'
        del self.z_layer  # z_layer is not needed anymore
        self.mu_layer, self.logvar_layer = None, None  # we use mu_layer and log var_layer instead of z_layer
        # Dropout, Encoder, and Decoder have been defined in AE class

        # defining mu_layer and log var_layer in VAE
        self.mu_layer = nn.Linear(encoder_dims[-1], z_dim)
        self.logvar_layer = nn.Linear(encoder_dims[-1], z_dim)

    #  to compute z based on mean (mu) and variance (log var) layers we built
    @staticmethod
    def reparameterize(mu, logvar):
        z = None
        eps = torch.normal(0, 1, size=list(mu.size()))
        z = mu + logvar.exp() * eps
        return z

    def forward(self, x):
        x_hat, mu, logvar = None, None, None
        dropped_out = self.dropout(x)
        temp = self.encoder(dropped_out)
        mu = self.mu_layer(temp)
        logvar = self.logvar_layer(temp)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def get_loss(self, x, x_hat, mu, logvar):
        MSE, KLD = 0, 0
        MSE = F.mse_loss(x, x_hat)
        # KL divergence loss
        KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu * mu - torch.exp(logvar), dim=1))
        return MSE + 20*KLD

    def generate(self, n):
        samples = None
        noise = torch.from_numpy(np.random.normal(0, 1, n)).float()
        samples = self.decoder(noise)
        return samples



