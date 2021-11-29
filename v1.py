# 
#   Deep Fusion
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import cat, chunk, linspace, meshgrid, stack, sum, Tensor
from torch.jit import export
from torch.nn import AdaptiveAvgPool2d, Conv2d, Module, ReLU, SELU, Sequential, Tanh
from torch.nn.functional import grid_sample, normalize
from torchsummary import summary
from typing import List

# Deep Bilateral Exposure Fusion

class DeepFusion (Module):

    def __init__ (self, exposure_count, grid_size):
        super(DeepFusion, self).__init__()
        self.grid_size = grid_size
        self.exposure_count = exposure_count
        # Bilateral grid construction network
        spatial_bins_x, spatial_bins_y, intensity_bins = self.grid_size
        self.grid_builder = Sequential( # NxCxHxW
            # In conv
            Conv2d(self.exposure_count * 3, 4 * self.exposure_count, kernel_size=7, stride=2, bias=True),
            SELU(inplace=True),
            # Level 1
            Conv2d(4 * self.exposure_count, 8 * self.exposure_count, kernel_size=3, stride=2, bias=True),
            SELU(inplace=True),
            Conv2d(8 * self.exposure_count, 8 * self.exposure_count, kernel_size=3, stride=1, bias=True),
            SELU(inplace=True),
            # Level 2
            Conv2d(8 * self.exposure_count, 16 * self.exposure_count, kernel_size=3, stride=2, bias=True),
            SELU(inplace=True),
            Conv2d(16 * self.exposure_count, 16 * self.exposure_count, kernel_size=3, stride=1, bias=True),
            SELU(inplace=True),
            # Level 3
            Conv2d(16 * self.exposure_count, 32 * self.exposure_count, kernel_size=3, stride=1, bias=True),
            SELU(inplace=True),
            Conv2d(32 * self.exposure_count, 32 * self.exposure_count, kernel_size=3, stride=1, bias=True),
            SELU(inplace=True),
            # Out conv
            Conv2d(32 * self.exposure_count, intensity_bins * self.exposure_count, kernel_size=3, stride=1, bias=True),
            ReLU(inplace=True), # Strict interpolation for fusion
            AdaptiveAvgPool2d((spatial_bins_x, spatial_bins_y))
        )
        # Guide map construction network
        # We use pointwise conv to construct the guide map from exposure stack
        self.guide_builder = Sequential(
            # In conv
            Conv2d(self.exposure_count * 3, 16, kernel_size=1, stride=1, padding=0, bias=True),
            SELU(inplace=True),
            # Out conv
            Conv2d(16, 1, kernel_size=1, stride=1, padding=0, bias=True),
            Tanh()
        )

    def forward (self, exposure_stack: Tensor) -> Tensor:
        weights = self.weight_maps(exposure_stack)
        fusion = self.fuse_exposures(exposure_stack, weights)
        return fusion

    def weight_maps (self, exposure_stack: Tensor) -> List[Tensor]:
        # Create a bilateral grid of coefficients
        # NxExIxSxS, where E is exposure count, I is intensity bins, S is spatial bins
        spatial_bins_x, spatial_bins_y, intensity_bins = self.grid_size
        batch_size = exposure_stack.shape[0]
        bilateral_grid = self.grid_builder(exposure_stack)
        bilateral_grid = bilateral_grid.view(batch_size, self.exposure_count, intensity_bins, spatial_bins_x, spatial_bins_y)
        bilateral_grid = normalize(bilateral_grid, p=1., dim=1)
        # Slice the bilateral grid for coefficients # NxExHxW
        guide_map = self.guide_builder(exposure_stack)
        coefficients = self.slice_bilateral_grid(bilateral_grid, guide_map)
        weights = chunk(coefficients, self.exposure_count, dim=1)
        return weights

    def fuse_exposures (self, exposure_stack: Tensor, weights: List[Tensor]) -> Tensor:
        # Weight exposures
        exposures = chunk(exposure_stack, self.exposure_count, dim=1)
        weighted_exposures = [exposure * weight for exposure, weight in zip(exposures, weights)]
        # Blend
        fusion = stack(weighted_exposures, dim=0).sum(dim=0)
        return fusion

    @export
    def guide_maps (self, exposure_stack: Tensor) -> List[Tensor]:
        guide_map = self.guide_builder(exposure_stack)
        return [guide_map]

    def slice_bilateral_grid (self, grid: Tensor, guide: Tensor): # `grid` is NxCxIxSxS in [0, inf), `guide` is Nx1xHxW in [-1., 1.]
        samples, _, height, width = guide.shape
        # Create slice grid
        hg, wg = meshgrid(linspace(-1., 1., height), linspace(-1., 1., width))
        hg = hg.repeat(samples, 1, 1).unsqueeze(dim=3).to(grid.device)
        wg = wg.repeat(samples, 1, 1).unsqueeze(dim=3).to(grid.device)
        slice_grid = guide.permute(0, 2, 3, 1).contiguous()     # NxHxWx1
        slice_grid = cat([wg, hg, slice_grid], dim=3)           # NxHxWx3
        slice_grid = slice_grid.unsqueeze(dim=1)                # Nx1xHxWx3
        # Sample
        result = grid_sample(grid, slice_grid, mode="bilinear", align_corners=False).squeeze(dim=2)
        return result


if __name__ == "__main__":
    EXPOSURE_COUNT = 3
    model = DeepFusion(exposure_count=EXPOSURE_COUNT, grid_size=(64, 64, 32))
    summary(model, (EXPOSURE_COUNT * 3, 512, 512), batch_size=8)