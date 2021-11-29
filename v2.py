# 
#   Deep Fusion
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import cat, chunk, linspace, meshgrid, split, stack, sum, tensor, Tensor
from torch.nn import AvgPool2d, Conv2d, Module, SELU, Sequential, Tanh, Upsample
from torch.nn.functional import grid_sample, softmax
from torchsummary import summary
from typing import List

class DeepFusionV2 (Module):
    """"
    Deep Bilateral Exposure Fusion V2.
    """

    def __init__ (self):
        super(DeepFusionV2, self).__init__()
        base_width = 16
        splat_width = 64
        self.splatting_block = Sequential(
            # Downsample
            Upsample(size=512, mode="bilinear", align_corners=False),
            # In conv
            Conv2d(3, base_width, kernel_size=7, dilation=2, bias=False),
            SELU(inplace=True),
            AvgPool2d(2),
            # Down conv 1
            Conv2d(base_width, 2 * base_width, kernel_size=3, dilation=2, bias=False),
            SELU(inplace=True),
            AvgPool2d(2),
            # Down conv 2
            Conv2d(2 * base_width, splat_width, kernel_size=3, dilation=2, bias=False),
            SELU(inplace=True),
            # Conv 1
            Conv2d(splat_width, splat_width, kernel_size=3, dilation=2, bias=False),
            SELU(inplace=True),
            # Conv 2
            Conv2d(splat_width, splat_width, kernel_size=3, dilation=4, bias=False),
            SELU(inplace=True),
            # Conv 3
            Conv2d(splat_width, splat_width, kernel_size=3, dilation=4, bias=False),
            SELU(inplace=True),
            # Conv 4
            Conv2d(splat_width, splat_width, kernel_size=3, dilation=4, bias=False),
            SELU(inplace=True),
            # Conv 5
            Conv2d(splat_width, splat_width, kernel_size=3, dilation=4, bias=False),
            SELU(inplace=True),
            # Out conv
            Conv2d(splat_width, 16, kernel_size=3, dilation=2, bias=False)
        )
        self.guide_block = Sequential(
            Conv2d(3, 16, kernel_size=1, bias=False),
            SELU(inplace=True),
            Conv2d(16, 1, kernel_size=1, bias=False),
            Tanh()
        )

    def forward (self, exposure_stack: Tensor) -> Tensor:
        # Compute weights
        weights = self.weight_maps(exposure_stack)
        fusion = self.fuse_exposures(exposure_stack, weights)
        return fusion

    def weight_maps (self, exposure_stack: Tensor) -> List[Tensor]:
        # Compute per-exposure weight grid
        exposures = split(exposure_stack, 3, dim=1)
        weight_grids = [self.splat_exposure(exposure) for exposure in exposures]
        # Slice weights from grid
        guide_maps = self.guide_maps(exposure_stack)
        weights = [self.slice_bilateral_grid(grid, guide) for grid, guide in zip(weight_grids, guide_maps)]
        # Normalize weights
        weights = cat(weights, dim=1)       # NxExHxW
        weights = softmax(weights, dim=1)   # Convert logits to probability density
        weights = split(weights, 1, dim=1)
        return weights

    def fuse_exposures (self, exposure_stack: Tensor, weights: List[Tensor]) -> Tensor:
        # Weight exposures
        exposures = split(exposure_stack, 3, dim=1)
        weighted_exposures = [exposure * weight for exposure, weight in zip(exposures, weights)]
        # Blend
        fusion = stack(weighted_exposures, dim=0).sum(dim=0)
        return fusion

    def guide_maps (self, exposure_stack: Tensor) -> List[Tensor]:
        exposures = split(exposure_stack, 3, dim=1)
        guide_maps = [self.guide_block(exposure) for exposure in exposures]
        return guide_maps

    def splat_exposure (self, exposure: Tensor) -> Tensor:
        bilateral_grid = self.splatting_block(exposure)
        bilateral_grid = bilateral_grid.unsqueeze(dim=1)
        return bilateral_grid

    def slice_bilateral_grid (self, grid: Tensor, guide: Tensor) -> Tensor: # `grid` is Nx1xIxSxS in [0, inf), `guide` is Nx1xHxW in [-1., 1.]
        samples, _, height, width = guide.shape
        # Create slice grid
        hg, wg = meshgrid(linspace(-1., 1., height), linspace(-1., 1., width))
        hg = hg.repeat(samples, 1, 1).unsqueeze(dim=3).to(grid.device)
        wg = wg.repeat(samples, 1, 1).unsqueeze(dim=3).to(grid.device)
        slice_grid = guide.permute(0, 2, 3, 1).contiguous()     # NxHxWx1
        slice_grid = cat([wg, hg, slice_grid], dim=3)           # NxHxWx3
        slice_grid = slice_grid.unsqueeze(dim=1)                # Nx1xHxWx3
        # Sample
        result = grid_sample(grid, slice_grid, mode="bilinear", padding_mode="border", align_corners=False).squeeze(dim=2)
        return result


if __name__ == "__main__":
    model = DeepFusionV2()
    summary(model, (9, 1024, 1024), batch_size=8)