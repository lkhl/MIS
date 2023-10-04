import gradio as gr
import torch

from .proposal import MaskProposalInterface
from .segmentation import InteractiveSegmentationInterface


_HEADER = """
<div style="text-align: center; margin: 10px">
    <h1> ⭐ <span style='color: #4D96FF;'>M</span><span style='color: #F9D923;'>I</span><span style='color: #EB5353;'>S</span>:
        <span style='color: #4D96FF;'>M</span>ulti-granularity <span style='color: #F9D923;'>I</span>nteraction <span style='color: #EB5353;'>S</span>imulation for Unsupervised Interactive Segmentation </h1>
</div>
<p align="center">
    <a href="https://arxiv.org/pdf/2303.13399.pdf"><img src="http://img.shields.io/badge/Paper-EB455F.svg?logo=arxiv" style="display:inline;"></a>
    <a href="https://lkhl.github.io/MIS"><img src="http://img.shields.io/badge/Project_Page-7149C6.svg?logo=openproject" style="display:inline;"></a>
    <a href="https://github.com/lkhl/MIS"><img src="https://img.shields.io/badge/Code-2B2A4C.svg?logo=github" style="display:inline;"></a>
    <a href="https://lkhl.github.io/MIS"><img src="https://img.shields.io/badge/Video-FC2947.svg?logo=youtube" style="display:inline;"></a>
    <a href="https://huggingface.co/spaces/lkhl/MIS?duplicate=true"><img src="https://img.shields.io/badge/Duplicate_This_Demo-EA906C?logo=buffer" style="display:inline;"></a>
</p>
"""


class MISWebApplication(object):

    def __init__(self, device: torch.device = None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with gr.Blocks() as self._blocks:
            gr.Markdown(_HEADER)
            with gr.Tab('Unsupervised Interactive Segmentation'):
                InteractiveSegmentationInterface(device=device)
            with gr.Tab('Multi-granularity Mask Proposal'):
                MaskProposalInterface(device=device)

    def launch(self):
        self._blocks.launch()
