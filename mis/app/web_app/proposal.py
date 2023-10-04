import gc
import random

import cv2
import gradio as gr
import numpy as np
import torch

from mis.model.feature_extractor import (DINOv1FeatureExtractor, DINOv2FeatureExtractor,
                                         MoCov3FeatureExtractor, build_transform)
from mis.ops import bottom_up_merging, get_leaf_descendents, get_roots
from mis.visualization.utils import draw_contour, draw_mask, draw_region, random_color_map


class MaskProposalInterface(object):

    def __init__(self, device: torch.device, max_regions: int = 50):
        self.device = device
        self.max_regions = max_regions

        self._pretrained_models = {
            'DINOv1': DINOv1FeatureExtractor,
            'DINOv2': DINOv2FeatureExtractor,
            'MoCov3': MoCov3FeatureExtractor
        }
        self._model_factory = None
        self._patch_size = None
        self._model = None
        self._transform = None

        self._img = None
        self._img_size = None
        self._feature_size = None
        self._tree = None
        self._region_id = None
        self._num_leaves = None
        self._color_map = None

        with gr.Blocks() as self._blocks:
            with gr.Tab('Input'):
                self._build_input_interface()
            with gr.Tab('Bottom-up'):
                self._build_merging_interface()
            with gr.Tab('Top-down'):
                self._build_splitting_interface()

        self._add_functions()

    def _build_input_interface(self):
        default_pretrained_model = list(self._pretrained_models.keys())[1]
        self._model_factory = self._pretrained_models[default_pretrained_model]
        available_model_size = self._model_factory.available_model_size
        available_patch_size = [str(x) for x in self._model_factory.available_patch_size]

        with gr.Row():
            self.input_img = gr.Image(type='pil', label='Input Image')
            with gr.Column():
                with gr.Row():
                    self.pretrained_model = gr.Dropdown(
                        choices=self._pretrained_models.keys(),
                        value=default_pretrained_model,
                        label='Pretrained Model',
                        interactive=True)
                    self.model_size = gr.Dropdown(
                        choices=available_model_size,
                        value=available_model_size[0],
                        label='Model Size',
                        interactive=True)
                    self.patch_size = gr.Dropdown(
                        choices=available_patch_size,
                        value=available_patch_size[0],
                        label='Patch Size',
                        interactive=True)
                with gr.Row():
                    self.loaded_model = gr.Textbox(label='Loaded Model', interactive=False)
                    self.load_button = gr.Button(value='Load Model', interactive=True)
                with gr.Row():
                    self.num_steps = gr.Textbox(label='Number of Merging Steps', interactive=False)
                    self.submit_button = gr.Button(value='Submit Image', interactive=False)

    def _build_merging_interface(self):
        self.merge_step = gr.Slider(step=1, label='Merge Step', interactive=False)
        with gr.Row():
            self.input_img_merging = gr.Image(label='Input Image', show_download_button=False)
            self.selected_region = gr.Image(label='Selected Regions', show_download_button=False)
            self.available_region = gr.Image(label='Available Regions', show_download_button=False)

    def _build_splitting_interface(self):
        with gr.Row():
            self.current_node_id = gr.Textbox(label='Current Node ID', interactive=False)
            self.left_child_id = gr.Textbox(label='Left Child ID', interactive=False)
            self.right_child_id = gr.Textbox(label='Right Child ID', interactive=False)
            self.reset_button = gr.Button(value='Reset', interactive=False)
            self.go_left_button = gr.Button(value='Go Left', interactive=False)
            self.go_right_button = gr.Button(value='Go Right', interactive=False)
        with gr.Row():
            self.input_img_splitting = gr.Image(label='Input Image', show_download_button=False)
            self.left_child = gr.Image(label='Left Child', show_download_button=False)
            self.right_child = gr.Image(label='Right Child', show_download_button=False)

    def _add_functions(self):
        self.input_img.upload(
            fn=self._load_image,
            inputs=self.input_img,
            outputs=[self.input_img_merging, self.input_img_splitting, self.submit_button])
        self.input_img.clear(fn=self._clear_image, outputs=self.submit_button)

        self.pretrained_model.change(
            fn=self._set_pretrained,
            inputs=self.pretrained_model,
            outputs=[self.model_size, self.patch_size])
        self.load_button.click(
            fn=self._load_model,
            inputs=[self.model_size, self.patch_size, self.input_img],
            outputs=[self.loaded_model, self.submit_button])

        self.submit_button.click(
            fn=self._preprocess_image,
            inputs=self.input_img,
            outputs=[
                self.num_steps, self.merge_step, self.reset_button, self.selected_region,
                self.available_region, self.current_node_id, self.left_child_id,
                self.right_child_id, self.left_child, self.right_child, self.go_left_button,
                self.go_right_button
            ])

        self.merge_step.release(
            fn=self._show_merged_regions,
            inputs=self.merge_step,
            outputs=[self.selected_region, self.available_region])

        self.reset_button.click(
            fn=self._reset_current_node,
            outputs=[
                self.current_node_id, self.left_child_id, self.right_child_id, self.left_child,
                self.right_child, self.go_left_button, self.go_right_button
            ])
        self.go_left_button.click(
            fn=self._set_current_node,
            inputs=self.left_child_id,
            outputs=[
                self.current_node_id, self.left_child_id, self.right_child_id, self.left_child,
                self.right_child, self.go_left_button, self.go_right_button
            ])
        self.go_right_button.click(
            fn=self._set_current_node,
            inputs=self.right_child_id,
            outputs=[
                self.current_node_id, self.left_child_id, self.right_child_id, self.left_child,
                self.right_child, self.go_left_button, self.go_right_button
            ])

    def _load_image(self, img):
        self._img = np.array(img)
        self._img_size = (img.height, img.width)
        return self._img, self._img, gr.update(interactive=self._model is not None)

    def _clear_image(self):
        return gr.update(interactive=False)

    def _set_pretrained(self, pretrained_model):
        self._model_factory = self._pretrained_models[pretrained_model]
        available_model_size = self._model_factory.available_model_size
        available_patch_size = [str(x) for x in self._model_factory.available_patch_size]
        return (gr.update(choices=available_model_size, value=available_model_size[0]),
                gr.update(choices=available_patch_size, value=available_patch_size[0]))

    def _load_model(self, model_size, patch_size, img):
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            torch.cuda.empty_cache()
        patch_size = int(patch_size)
        self._patch_size = patch_size
        self._model = self._model_factory(model_size, patch_size)
        self._model.eval()
        self._model.to(self.device)
        self._transform = build_transform(scale_factor=patch_size / 8.0, patch_size=patch_size)
        loaded_model = f'{self._model_factory.__name__}({model_size}, {patch_size})'
        return loaded_model, gr.update(interactive=img is not None)

    @torch.no_grad()
    def _preprocess_image(self, img):
        img = self._transform(img).unsqueeze(0).to(self.device)
        self._feature_size = (img.size(-2) // self._patch_size, img.size(-1) // self._patch_size)
        features = self._model(img)[0].cpu().numpy()
        self._tree = bottom_up_merging(features, *self._feature_size)
        self._region_id = get_roots(self._tree).reshape(-1, *self._feature_size).astype(np.uint16)
        self._num_leaves = len(self._tree) + 1
        self._color_map = random_color_map(2 * self._num_leaves - 1)
        minimum = max(1, self._num_leaves - self.max_regions)
        maximum = self._num_leaves - 1
        value = random.randint(minimum, maximum)
        return (maximum, gr.update(minimum=minimum, maximum=maximum, value=value,
                                   interactive=True), gr.update(interactive=True),
                *self._show_merged_regions(value), *self._reset_current_node())

    def _show_merged_regions(self, merge_step):
        merge_id = self._tree[merge_step - 1]
        merge_mask_1, merge_mask_2 = self._region_id[
            merge_step - 1] == merge_id[0], self._region_id[merge_step - 1] == merge_id[1]
        merge_mask_1 = cv2.resize(
            merge_mask_1.astype(np.uint8), self._img_size[::-1], interpolation=cv2.INTER_NEAREST)
        merge_mask_2 = cv2.resize(
            merge_mask_2.astype(np.uint8), self._img_size[::-1], interpolation=cv2.INTER_NEAREST)
        merge_mask = np.logical_or(merge_mask_1, merge_mask_2)

        last_region_mask = cv2.resize(
            self._region_id[merge_step - 1], self._img_size[::-1], interpolation=cv2.INTER_NEAREST)
        last_masked_img = draw_region(self._img, last_region_mask, self._color_map, opacity=0.5)
        last_hilighted_img = draw_mask(last_masked_img, merge_mask)
        last_hilighted_img = draw_contour(last_hilighted_img, merge_mask_1, color=(255, 255, 255))
        last_hilighted_img = draw_contour(last_hilighted_img, merge_mask_2, color=(255, 255, 255))

        region_mask = cv2.resize(
            self._region_id[merge_step], self._img_size[::-1], interpolation=cv2.INTER_NEAREST)
        masked_img = draw_region(self._img, region_mask, self._color_map, opacity=0.5)

        return last_hilighted_img, masked_img

    def _get_mask(self, node):
        mask = np.zeros(self._num_leaves, dtype=np.uint8)
        mask[get_leaf_descendents(self._tree, node)] = 1
        mask = mask.reshape(*self._feature_size)
        mask = cv2.resize(mask, self._img_size[::-1], interpolation=cv2.INTER_NEAREST)
        img = draw_mask(self._img, mask, opacity=0.7)
        img = draw_contour(img, mask, color=(255, 255, 255))
        return img

    def _set_current_node(self, cur_node):
        cur_node = int(cur_node)
        left_node = self._tree[cur_node - self._num_leaves][0]
        right_node = self._tree[cur_node - self._num_leaves][1]
        go_left = gr.update(interactive=left_node >= self._num_leaves)
        go_right = gr.update(interactive=right_node >= self._num_leaves)
        left_mask = self._get_mask(left_node)
        right_mask = self._get_mask(right_node)
        return cur_node, left_node, right_node, left_mask, right_mask, go_left, go_right

    def _reset_current_node(self):
        return self._set_current_node(self._num_leaves * 2 - 2)
