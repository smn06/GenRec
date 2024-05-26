import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from gan_model import Generator, Discriminator

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data.values

def train_gan(generator, discriminator, dataloader, latent_dim, num_epochs, device):
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for i, real_samples in enumerate(dataloader):
            real_samples = real_samples.to(device)
            batch_size = real_samples.size(0)

            # Labels
            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim).to(device)
            generated_samples = generator(z)
            g_loss = criterion(discriminator(generated_samples), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(real_samples), valid)
            fake_loss = criterion(discriminator(generated_samples.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

    return generator, discriminator

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train GAN for generating synthetic user profiles.')
    parser.add_argument('--data', type=str, required=True, help='Path to real user profiles data')
    parser.add_argument('--latent_dim', type=int, default=100, help='Dimensionality of latent space')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10000, help='Number of training epochs')
    args = parser.parse_args()

    # Load data
    real_profiles = load_data(args.data)
    real_profiles = (real_profiles - real_profiles.min()) / (real_profiles.max() - real_profiles.min()) * 2 - 1  # Normalize to [-1, 1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(TensorDataset(torch.tensor(real_profiles, dtype=torch.float32)), batch_size=args.batch_size, shuffle=True)

    # Initialize models
    generator = Generator(args.latent_dim, real_profiles.shape[1]).to(device)
    discriminator = Discriminator(real_profiles.shape[1]).to(device)

    # Train GAN
    trained_generator, trained_discriminator = train_gan(generator, discriminator, dataloader, args.latent_dim, args.num_epochs, device)

    # Save models
    torch.save(trained_generator.state_dict(), 'generator.pth')
    torch.save(trained_discriminator.state_dict(), 'discriminator.pth')
