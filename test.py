# 
#   Deep Fusion
#   Copyright (c) 2020 Homedeck, LLC.
#

from argparse import ArgumentParser
from PIL import Image
from torch import cat, device as get_device, set_grad_enabled
from torch.cuda import is_available as cuda_available
from torch.jit import load
from torchvision.transforms import Compose, Normalize, ToPILImage, ToTensor

# Parse arguments
parser = ArgumentParser(description="Deep Fusion: Test")
parser.add_argument("--model", type=str, default="deep_fusion.pt", help="Path to trained model")
parser.add_argument("--exposures", type=str, nargs="+", help="Path to exposures")
args = parser.parse_args()

# Load model
device = get_device("cuda:0") if cuda_available() else get_device("cpu")
model = load(args.model, map_location=device).to(device)
set_grad_enabled(False)

# Load exposures
to_tensor = Compose([
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
exposures = [Image.open(path) for path in args.exposures]
exposures = [to_tensor(exposure) for exposure in exposures]
exposure_stack = cat(exposures, dim=0).unsqueeze(dim=0)

# Run inference
fusion = model(exposure_stack)
weights = model.weight_maps(exposure_stack)
guides = model.guide_maps(exposure_stack)

# Write results
Compose([
    Normalize(mean=[-1., -1., -1.], std=[2., 2., 2.]),
    ToPILImage()
])(fusion.squeeze()).save("fusion.jpg")
for i, weight in enumerate(weights):
    ToPILImage()(weight.squeeze(dim=0)).save(f"weight_{i}.jpg")
for i, guide in enumerate(guides):
    ToPILImage()((-guide.squeeze(dim=0) + 1.) / 2.).save(f"guide_{i}.jpg")