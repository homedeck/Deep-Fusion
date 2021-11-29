# 
#   Deep Fusion
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import clamp, norm, std, tensor, Tensor
from torch.nn import Module
from torch.nn.functional import conv2d, pad

class ContrastLoss (Module):
    """
    Contrast loss, from Mertens et al.
    """

    def __init__ (self):
        super(ContrastLoss, self).__init__()
        # Gaussian kernel
        gaussian_kernel = Tensor([
            [1., 4., 6., 4., 1.],
            [4., 16., 24., 16., 4.],
            [6., 24., 36., 24., 6.],
            [4., 16., 24., 16., 4.],
            [1., 4., 6., 4., 1.]
        ])
        gaussian_kernel /= 16.
        gaussian_kernel = gaussian_kernel.view(1, 1, 5, 5).repeat(3, 1, 1, 1)
        # Laplacian kernel
        laplacian_kernel = Tensor([
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., -24., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.]
        ])
        laplacian_kernel = laplacian_kernel.view(1, 1, 5, 5).repeat(3, 1, 1, 1)
        # Register
        self.register_buffer("gaussian_kernel", gaussian_kernel)
        self.register_buffer("laplacian_kernel", laplacian_kernel)

    def forward (self, input: Tensor, target: Tensor):
        input_laplacian = self.__laplacian(input)
        target_laplacian = self.__laplacian(target)
        delta = clamp(target_laplacian - input_laplacian, min=0)
        loss = delta.sum() / delta.nelement()
        return loss

    def __laplacian (self, input: Tensor):
        # Denoise with Gaussian
        input = pad(input, (2, 2, 2, 2), mode="reflect")
        input = conv2d(input, self.gaussian_kernel, groups=3)
        # Take Laplacian
        input = pad(input, (2, 2, 2, 2), mode="reflect")
        laplacian = conv2d(input, self.laplacian_kernel, groups=3)
        # Get absolute response
        response = laplacian.abs()
        return response


class SaturationLoss (Module):
    """
    Saturation loss, from Mertens et al.
    """

    def __init__ (self):
        super(SaturationLoss, self).__init__()

    def forward (self, input: Tensor, target: Tensor):
        input_uv = self.__rgb_to_yuv(input)[:,1:,:,:]
        target_uv = self.__rgb_to_yuv(target)[:,1:,:,:]
        input_sat, target_sat = norm(input_uv, dim=1), norm(target_uv, dim=1)
        delta = clamp(target_sat - input_sat, min=0.)
        loss = delta.sum() / delta.nelement()
        return loss

    def __rgb_to_yuv (self, input: Tensor) -> Tensor: # from Deep Color
        RGB_TO_YUV = tensor([
            [0.2126, 0.7152, 0.0722],
            [-0.09991, -0.33609, 0.436],
            [0.615, -0.55861, -0.05639]
        ]).float().to(input.device)
        input = (input + 1.) / 2.
        rgb_colors = input.flatten(start_dim=2)
        yuv_colors = RGB_TO_YUV.matmul(rgb_colors)
        yuv = yuv_colors.view_as(input)
        return yuv