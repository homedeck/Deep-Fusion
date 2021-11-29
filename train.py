# 
#   Deep Fusion
#   Copyright (c) 2020 Homedeck, LLC.
#

from argparse import ArgumentParser
from colorama import Fore, Style
from suya import set_suya_access_key
from torch import device as get_device, rand
from torch.cuda import is_available as cuda_available
from torch.jit import save, script
from torch.nn import L1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchsummary import summary
import tableprint

from dataset import FusionDataset
from loss import ContrastLoss,SaturationLoss
from v2 import DeepFusionV2

# Parse arguments
parser = ArgumentParser(description="Deep Fusion: Training")
parser.add_argument("--suya-key", type=str, required=False, default=None, help="Suya access key")
parser.add_argument("--tag", type=str, required=True, help="Dataset tag on Suya")
parser.add_argument("--learning-rate", type=float, default=2e-5, help="Nominal learning rate")
parser.add_argument("--epochs", type=int, default=50, help="Epochs")
parser.add_argument("--lambda-contrast", type=float, default=0.08, help="Contrast loss regularization weight")
parser.add_argument("--lambda-saturation", type=float, default=4.0, help="Saturation loss regularization weight")
parser.add_argument("--batch-size", type=int, default=8, help="Minibatch size")
parser.add_argument("--patch-size", type=int, default=1024, help="Patch size")
args = parser.parse_args()

# Create dataset
set_suya_access_key(args.suya_key)
dataset = FusionDataset(args.tag, dataset_size=1000, patch_size=args.patch_size)
dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, drop_last=True, pin_memory=True, shuffle=True)

# Create model
device = get_device("cuda:0") if cuda_available() else get_device("cpu")
model = DeepFusionV2().to(device)

# Create losses
content_loss = L1Loss().to(device)
contrast_loss = ContrastLoss().to(device)
saturation_loss = SaturationLoss().to(device)

# Create optimizer
optimizer = Adam(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

# Print
print("Preparing for training:")
summary(model, (9, args.patch_size, args.patch_size), batch_size=args.batch_size)

# Create summary writer
with SummaryWriter() as summary_writer:

    # Print table and graph
    HEADERS = ["Iteration", "Epoch", "Content"]
    print(tableprint.header(HEADERS))

    # Setup for training
    model.train(mode=True)
    iteration_index = 0
    last_loss = 1e+10

    # Train
    for epoch in range(args.epochs):

        # Iterate over all minibatches
        for exposure_stack, fusion in dataloader:

            # Run forward pass
            exposure_stack, fusion = exposure_stack.to(device), fusion.to(device)
            prediction = model(exposure_stack)       

            # Compute losses
            loss_content = content_loss(prediction, fusion)
            loss_contrast = contrast_loss(prediction, fusion)
            loss_saturation = saturation_loss(prediction, fusion)
            loss_total = loss_content + args.lambda_contrast * loss_contrast + args.lambda_saturation * loss_saturation

            # Backpropagate
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            # Log
            summary_writer.add_scalar("Deep Fusion/Content Loss", loss_content, iteration_index)
            summary_writer.add_scalar("Deep Fusion/Contrast Loss", loss_contrast, iteration_index)
            summary_writer.add_scalar("Deep Fusion/Saturation Loss", loss_saturation, iteration_index)
            summary_writer.add_scalar("Deep Fusion/Total Loss", loss_total, iteration_index)
            print(tableprint.row([
                f"{iteration_index}",
                f"{epoch}",
                f"{Style.BRIGHT}{Fore.GREEN if loss_total < last_loss else Fore.RED}{loss_total:.4f}{Style.RESET_ALL}"
            ]))
            last_loss = loss_total
            iteration_index += 1

        # Log images
        to_grid = lambda mbatch: make_grid(mbatch.cpu(), range=(-1., 1.), normalize=True)
        summary_writer.add_image("Input", to_grid(exposure_stack[:,:3,:,:]), iteration_index)
        summary_writer.add_image("Prediction", to_grid(prediction), iteration_index)
        summary_writer.add_image("Target", to_grid(fusion), iteration_index)

        # Save model
        model.cpu()
        scripted_model = script(model)
        save(scripted_model, "deep_fusion.pt")
        if "cuda" in str(device):
            model.cuda()

    # Print
    print(tableprint.bottom(len(HEADERS)))