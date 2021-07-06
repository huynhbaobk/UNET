from os import device_encoding
from utils import get_loaders, save_checkpoint
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_to_imgs,
)

# Hyperparameters

root = '/content/gdrive/MyDrive/KNG_Project/Windows/UNET'
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 120
NUM_WORKERS = 2
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "/content/windows/rs_224/images"
TRAIN_MASK_DIR = "/content/windows/rs_224/labels"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = np.transpose(targets, [0, 3, 1, 2]).float().to(device=DEVICE)
        # Forward
        print(targets.shape)
        with torch.cuda.amp.autocast():
            predictions = model(data)
            print(predictions.shape)
            # predictions = predictions.view(-1, IMAGE_HEIGHT*IMAGE_WIDTH*4)
            # predictions = predictions.unsqueeze(1).unsqueeze(-1)
            loss = torch.sqrt(loss_fn(predictions, targets))

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            # A.Rotate(limit=35, p=1.0),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.1),
            # A.Normalize(
            #     mean = [0.0, 0.0, 0.0],
            #     std = [1.0, 1.0, 1.0],
            #     max_pixel_value = 255.0,
            # ),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format='xy'))

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            # A.Normalize(
            #     mean = [0.0, 0.0, 0.0],
            #     std = [1.0, 1.0, 1.0],
            #     max_pixel_value = 255.0,
            # ),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format='xy'))

    model = UNET(in_channels=3, out_channels=4).to(device=DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        random_seed=42,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print("[INFO] Epoch: ", epoch)
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        #save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict() 
        }
        save_checkpoint(checkpoint, filename=root + '/my_checkpoint.pth.tar')

    #Save predictions
    # save_predictions_to_imgs(
    #     val_loader, model, folder=root+'/saved_images/', device=DEVICE
    # )

if __name__ == "__main__":
    main()
