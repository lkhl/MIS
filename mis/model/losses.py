import torch
import torch.nn as nn
import torch.nn.functional as F


def unfold(x, kernel_size=3, dilation=1):
    assert x.dim() == 4
    assert kernel_size % 2 == 1
    N, C, H, W = x.shape

    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    if padding > 0:
        x = F.pad(x, (padding, padding, padding, padding), mode='replicate')

    unfolded_x = F.unfold(x, kernel_size=kernel_size, dilation=dilation)
    unfolded_x = unfolded_x.reshape(N, C, -1, H, W)

    # remove the center pixels
    size = kernel_size**2
    unfolded_x = torch.cat((unfolded_x[:, :, :size // 2], unfolded_x[:, :, size // 2 + 1:]), dim=2)

    return unfolded_x


class SmoothLoss(nn.Module):

    def __init__(self, kernel_size=3, dilation=2, sigma_spatial=16, sigma_luma=16, sigma_chroma=8):
        super(SmoothLoss, self).__init__()
        self.kerenl_size = kernel_size
        self.dilation = dilation
        self.sigma_spatial = sigma_spatial
        self.sigma_luma = sigma_luma
        self.sigma_chroma = sigma_chroma

        yuv = torch.as_tensor([[0.299, 0.587, 0.114], [-0.147, -0.289, 0.436],
                               [0.615, -0.515, -0.100]])
        self.register_buffer('_yuv', yuv, persistent=False)

    def _compute_bilateral_affinity(self, img):
        img = torch.einsum('nchw, ck -> nkhw', img, self._yuv.to(img.device))
        pos = torch.stack(
            torch.meshgrid(torch.arange(img.size(2)), torch.arange(img.size(3))), dim=0)
        pos = pos.unsqueeze(0).to(img.dtype).to(img.device)

        unfolded_img = unfold(img, kernel_size=self.kerenl_size, dilation=self.dilation)
        unfolded_pos = unfold(pos, kernel_size=self.kerenl_size, dilation=self.dilation)

        img_diff = img[:, :, None] - unfolded_img
        pos_diff = pos[:, :, None] - unfolded_pos

        pos_term = pos_diff.pow(2).sum(1) / self.sigma_spatial / 2.0
        luma_term = img_diff[:, 0].pow(2) / self.sigma_luma / 2.0
        chroma_term = img_diff[:, 1:].pow(2).sum(1) / self.sigma_chroma / 2.0

        return torch.exp(-pos_term - luma_term - chroma_term)

    def forward(self, pred, target, image):
        image = F.interpolate(image, size=pred.shape[-2:], mode='bilinear', align_corners=False)
        affinity = self._compute_bilateral_affinity(image)

        unfolded_pred = unfold(pred, kernel_size=self.kerenl_size, dilation=self.dilation)
        pred_diff = pred[:, :, None] - unfolded_pred

        losses = affinity * pred_diff[:, 0].pow(2)

        return losses.mean()


class BCEWithLogitsLoss(nn.Module):

    def forward(self, pred, target, image=None):
        return F.binary_cross_entropy_with_logits(pred, target)
