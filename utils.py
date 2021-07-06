import torch
import torchvision
from dataset import WindowsDataset
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Save checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    keypoints_dir,
    batch_size,
    train_transform,
    val_transform,
    random_seed,
    valid_size=0.2,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
):
    train_ds = WindowsDataset(
        images_dir=train_dir,
        keypoints_dir=keypoints_dir,
        transforms=train_transform
    )

    val_ds = WindowsDataset(
        images_dir=train_dir,
        keypoints_dir=keypoints_dir,
        transforms=val_transform
    )

    num_train = len(train_ds)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixel = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds==y).sum()
            num_pixel += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
                )
                
    model.train()
    print (
        f"Got {num_correct}/{num_pixel} with acc {num_correct/num_pixel*100:.2f}"
    )

    print (f"Dice score: {dice_score/len(loader)}")

def save_predictions_to_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sum(model(x), dim=1)
            preds = (preds > 0.5).float()
        
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/truth_{idx}.png")
    
    model.train()

def predict_image(model, image_path):
    model.eval()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BRG2RGB)
    image = torch.unsqueeze(image, 0)

    with torch.no_grad():
        x = x.to(device)
        preds = model(x)
        preds = preds.detach().cpu().numpy()

    model.train()