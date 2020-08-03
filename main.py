import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
import argparse
from tqdm import tqdm
from dataset import ImagesDataset
from model import Generator, Discriminator 

def data_preprocessing(batch_size,download=False):
    img_size = 128
    data_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = ImagesDataset('./data',data_transforms)
    train_size = len(dataset)*90//100
    test_size = 70
    val_size = len(dataset)-train_size-test_size
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])
    dataloaders = {'train': torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True),
                   'val': torch.utils.data.DataLoader(val_set, batch_size=batch_size,shuffle=True)
                }
    dataset_sizes = {'train': train_size, 'val': val_size}
    return dataloaders,test_set,dataset_sizes


def visualize_images(images,save_image=False,image_name='images.jpg'):
    fig,ax = plt.subplots(figsize=(15,15))
    ax.set_xticks([]); ax.set_yticks([])
    grid = torchvision.utils.make_grid(images,10)
    if save_image:
        torchvision.utils.save_image(grid, f'imgs/{image_name}')
    ax.imshow(grid.cpu().permute([1,2,0]).detach().numpy())
    plt.show()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train(dataset,dataset_sizes,coding_size,generator,discriminator,device,epochs,opt_d,opt_g,criterion,schedulerD=None,schedulerG=None):
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        for phase in ['train','val']:
            if phase == 'train':
                generator.train()
                discriminator.train()
            else:
                generator.eval()
                discriminator.eval()

            losses_d = 0
            losses_g = 0
            for X_batch in tqdm(dataset[phase]):
                X_batch = X_batch.to(device)
                batch_size = X_batch.size(0)

                # Training Discriminator      
                y_real = torch.ones(batch_size).to(device)
                y_fake = torch.zeros(batch_size).to(device)
                
                opt_d.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    for param in discriminator.parameters():
                        param.required_grad=True
                    d_output = discriminator(X_batch).view(-1,)
                    real_loss_d = criterion(d_output, y_real)

                    fake = torch.randn(batch_size,coding_size).to(device)
                    generated_images = generator(fake)
                    
                    d_output = discriminator(generated_images).view(-1,)
                    fake_loss_d = criterion(d_output, y_fake)

                    loss_d = fake_loss_d + real_loss_d

                    if phase=='train':
                        loss_d.backward()
                        opt_d.step()

                losses_d += loss_d.item()*batch_size

                # Training Generator
                fake = torch.randn(batch_size,coding_size).to(device)
                y2 = torch.ones(batch_size).to(device)
                    
                opt_g.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    for param in discriminator.parameters():
                        param.required_grad=False
                    g_output = generator(fake)
                    d_output = discriminator(g_output).view(-1,)
                    loss_g = criterion(d_output,y2)

                    if phase=='train':
                        loss_g.backward()
                        opt_g.step()

                losses_g += loss_g.item()*batch_size


            if schedulerD!=None and phase=='train':
                schedulerD.step()                

            if schedulerG!=None and phase=='train':
                schedulerG.step()                

            epoch_loss_d = losses_d/dataset_sizes[phase]
            epoch_loss_g = losses_g/dataset_sizes[phase]
       

            print(f'{phase.capitalize()} D_Loss: {epoch_loss_d:.4f}, G_LOSS: {epoch_loss_g:.4f}')
            print()


def main():

    parser = argparse.ArgumentParser(description='Generative adversarial network')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                        help='learning rate (default: 2e-4)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')


    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    use_cuda = True
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")  
    coding_size = 500
    dataloaders,test_set,dataset_sizes = data_preprocessing(args.batch_size,False)

    generator = Generator(coding_size).to(device)
    discriminator = Discriminator().to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    optimizerD = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    train(dataloaders,dataset_sizes,coding_size,generator,discriminator,device,args.epochs,optimizerD,optimizerG,criterion)

    noise = torch.randn(10,coding_size).to(device)
    generated_imgs = generator(noise)
    visualize_images(generated_imgs,True)

if __name__ == '__main__':
    main()
   
