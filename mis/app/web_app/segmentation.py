import gc

import gradio as gr
import numpy as np
import torch

from isegm.inference.clicker import Click, Clicker
from isegm.inference.predictors import BasePredictor
from isegm.inference.transforms import ZoomIn
from isegm.inference.utils import load_single_is_model
from mis.visualization.utils import draw_click, draw_contour, draw_mask


class InteractiveSegmentationInterface(object):

    def __init__(self, device: torch.device):
        self.device = device

        self._clicker = Clicker()

        self._pretrained_models = {
            'mis_simpleclick_vit-b': './weights/mis_simpleclick_base448_sbd.pth'
        }
        self._predictor = None

        self._pred_prob = None
        self._masked_img = None

        self._build_interface()
        self._add_functions()

    def _build_interface(self):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    choices = list(self._pretrained_models.keys())
                    self.model_name = gr.Dropdown(choices=choices, value=choices[0], label='Model')
                    self.loaded_model = gr.Textbox(label='Loaded Model', interactive=False)
                    self.load_button = gr.Button(value='Load Model')
                with gr.Row():
                    self.input_img = gr.Image(label='Input Image')
                    self.click_map = gr.Image(
                        label='Click Map', show_download_button=False, interactive=False)

                with gr.Row():
                    self.add_button = gr.Button(value='Add Click', interactive=False)
                    self.undo_button = gr.Button(value='Undo', interactive=False)
                    self.submit_button = gr.Button(value='Segment', interactive=False)

                self.drawing_board = gr.Image(
                    label='Add Click',
                    tool='sketch',
                    interactive=False,
                    visible=False,
                    brush_radius=15)
                with gr.Row():
                    self.pos_button = gr.Button(value='Add Positive', visible=False)
                    self.neg_button = gr.Button(value='Add Negative', visible=False)
                    self.cancel_button = gr.Button(value='Cancel', visible=False)

            with gr.Column():
                self.threshold = gr.Slider(
                    label='Threshold',
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.01,
                    interactive=False)
                self.seg_mask = gr.Image(
                    label='Segmentation', show_download_button=False, interactive=False)

    def _add_functions(self):
        self.input_img.upload(
            fn=self._load_image,
            inputs=self.input_img,
            outputs=[
                self.click_map, self.seg_mask, self.add_button, self.undo_button,
                self.submit_button, self.threshold, self.drawing_board, self.pos_button,
                self.neg_button, self.cancel_button
            ])

        self.load_button.click(
            fn=self._load_model,
            inputs=[self.model_name, self.input_img],
            outputs=[self.loaded_model, self.submit_button])

        self.add_button.click(
            fn=self._create_click,
            outputs=[self.drawing_board, self.pos_button, self.neg_button, self.cancel_button])
        self.undo_button.click(
            fn=self._undo_click,
            outputs=[self.click_map, self.drawing_board, self.undo_button, self.submit_button])

        self.pos_button.click(
            fn=self._add_pos_click,
            inputs=self.drawing_board,
            outputs=[
                self.click_map, self.undo_button, self.submit_button, self.drawing_board,
                self.pos_button, self.neg_button, self.cancel_button
            ])
        self.neg_button.click(
            fn=self._add_neg_click,
            inputs=self.drawing_board,
            outputs=[
                self.click_map, self.undo_button, self.submit_button, self.drawing_board,
                self.pos_button, self.neg_button, self.cancel_button
            ])
        self.cancel_button.click(
            fn=self._cancel,
            outputs=[self.drawing_board, self.pos_button, self.neg_button, self.cancel_button])

        self.submit_button.click(
            fn=self._segment,
            inputs=[self.input_img, self.threshold],
            outputs=[self.seg_mask, self.click_map, self.drawing_board, self.threshold])
        self.threshold.release(
            fn=self._show_mask,
            inputs=self.threshold,
            outputs=[self.seg_mask, self.click_map, self.drawing_board])

    @property
    def _click_map(self):
        if self._img is None:
            return None
        img = self._img if self._masked_img is None else self._masked_img
        return draw_click(img, self._clicker.get_clicks())

    def _load_image(self, img):
        self._img = img
        self._img_size = img.shape[:2]
        self._clicker.reset_clicks()
        self._pred_prob = None
        self._masked_img = None
        return (self._click_map, None, gr.update(interactive=True), gr.update(interactive=False),
                gr.update(interactive=False), gr.update(interactive=False), *self._cancel())

    def _load_model(self, model_name, img):
        if self._predictor is not None:
            del self._predictor
            self._predictor = None
            gc.collect()
            torch.cuda.empty_cache()
        # state_dict = torch.hub.load_state_dict_from_url(model_name, map_location='cpu')
        state_dict = torch.load(self._pretrained_models[model_name], map_location='cpu')
        model = load_single_is_model(state_dict, device=self.device, eval_ritm=False)
        zoom_in = ZoomIn(skip_clicks=-1, target_size=(448, 448))
        self._predictor = BasePredictor(model, device=self.device, zoom_in=zoom_in, with_flip=True)
        enable_submit = img is not None and len(self._clicker) > 0
        return model_name, gr.update(interactive=enable_submit)

    def _create_click(self):
        return gr.update(
            value=self._click_map, interactive=True,
            visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

    def _cancel(self):
        return gr.update(
            interactive=False, visible=False), gr.update(visible=False), gr.update(
                visible=False), gr.update(visible=False)

    def _add_click(self, inp, is_positive):
        coords = np.nonzero(inp['mask'].sum(axis=-1))
        if len(coords[0]) == 0:
            return (self._click_map, gr.update(interactive=False), gr.update(interactive=False),
                    *self._cancel())
        coords = (round(coords[0].mean()), round(coords[1].mean()))
        click = Click(is_positive=is_positive, coords=coords)
        self._clicker.add_click(click)
        return (self._click_map, gr.update(interactive=True),
                gr.update(interactive=self._predictor is not None), *self._cancel())

    def _add_pos_click(self, inp):
        return self._add_click(inp, is_positive=True)

    def _add_neg_click(self, inp):
        return self._add_click(inp, is_positive=False)

    def _undo_click(self):
        self._clicker._remove_last_click()
        has_clicks = len(self._clicker) > 0
        click_map = self._click_map
        return (
            click_map,
            click_map,
            gr.update(interactive=has_clicks),
            gr.update(interactive=has_clicks),
        )

    @torch.no_grad()
    def _segment(self, img, threshold):
        self._predictor.set_input_image(img)
        self._pred_prob = self._predictor.get_prediction(self._clicker)
        return (*self._show_mask(threshold), gr.update(value=0.5, interactive=True))

    def _show_mask(self, threshold):
        mask = self._pred_prob > threshold
        img = draw_mask(self._img, mask)
        img = draw_contour(img, mask)
        self._masked_img = img
        click_map = self._click_map
        return img, click_map, click_map
