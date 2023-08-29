import os
import argparse
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision.transforms import transforms
from torchsummary import summary
# from torchmetrics import JaccardIndex

from dataset import ImagePairDataset
from model.unet import UNet
from model.unet2plus import UNet2Plus

# Conditional main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training script.')
    parser.add_argument( '--path', '-p',
                         help = 'path to data folder',
                         default = './data/',
                         type = str )
    parser.add_argument( '--model', '-m',
                         help = 'model to use (unet, unet2plus, or unet3plus)',
                         default = 'unet',
                         type = str )
    parser.add_argument( '--epochs',
                         help = 'numer of training epochs',
                         default = 100,
                         type = int )
    parser.add_argument( '--batch_size',
                         help = 'batch size for training dataset',
                         default = 4,
                         type = int )
    parser.add_argument( '--num_workers',
                         help = 'number of paralell workers',
                         default = 4,
                         type = int )
    parser.add_argument( '--normalize', '-n', 
                         action = 'store_true', default=False,
                         help = 'normalize dataset')
    args = parser.parse_args()
    
    
    # Transform with normalization (or not)
    if args.normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
            ])
    else:
        transform = transforms.ToTensor()
    
    try:
    
        # Load dataset
        dataset = ImagePairDataset(os.path.join(args.path, 'data'), transform = transform)
        
        # Split dataset (80/20)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        # Report split sizes
        print('Training dataset has {} instances'.format(len(train_set)))
        print('Validation dataset has {} instances'.format(len(val_set)))

        # Creat instances of dataloaders
        training_loader = DataLoader(train_set, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers, pin_memory = True)
        validation_loader = DataLoader(val_set, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers, pin_memory = True)
    
    except FileNotFoundError:
        print('[Error] Could not load dataset.')
        print('[Error] Make sure that you have annotated data for image files in path: {}'.format(args.path))

    # Get the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training model '{}' on device '{}'".format(args.model, device))

    # Create an instance of the U-Net model 
    in_channels = 3   # Number of input channels (e.g., RGB image)
    out_channels = 1  # Number of output channels (e.g., segmentation mask)
    if args.model == 'unet2plus':
        model = UNet2Plus(in_channels, out_channels)
    else:
        model = UNet(in_channels, out_channels)
    model.to(device)
    summary(model, (3, 256, 256))

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    # criterion = JaccardIndex(task='multiclass', num_classes=2)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    # Ensure directory for storing checkpoints and losses
    checkpoint_path = os.path.join(args.path, 'checkpoints')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    loss_path = os.path.join(args.path, 'losses')
    if not os.path.exists(loss_path):
        os.makedirs(loss_path)

    '''
    # Load model
    checkpoint = torch.load(os.path.join(checkpoint_path, f"{args.model}_best.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    # model.eval() for inference, or
    # model.train() for contunie training
    '''

    # Train loop
    losses = {'loss': [], 'epoch': []}
    best_loss = float('inf')  # Initialize with a high value                                   
    for epoch in range(args.epochs):

        # Traning
        model.train()
        train_loss = 0.0
        for inputs, targets in tqdm((training_loader)):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Zero parameter gradients
            optimizer.zero_grad()
            
            # Forward pass and compute loss
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_loss += loss.item()

            # Backward pass and optimization    
            loss.backward()
            optimizer.step()

        # Compute average training loss                      
        train_loss /= len(training_loader)

        # Print training loss for each epoch
        print(f"Epoch {epoch + 1}/{args.epochs}: Train Loss: {train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(validation_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass and compute loss               
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        # Compute average validation loss
        val_loss /= len(validation_loader)

        # Print validation loss for each epoch
        print(f"Epoch {epoch + 1}/{args.epochs}: Val Loss: {val_loss:.4f}")
        losses['loss'].append(val_loss)
        losses['epoch'].append(epoch + 1)
        with open(os.path.join(loss_path, f"{args.model}_losses.json"), 'w') as f:
            json.dump(losses, f, indent = 2)

        # Save the best model checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(checkpoint_path, f"{args.model}_best.pt") 
            )

        # Save save checkpoints every 100th epcoh
        if (epoch + 1) % 100 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(checkpoint_path, f"{args.model}_epoch_{epoch + 1}.pt") 
            )
