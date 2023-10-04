import os
import os.path as osp

import cv2
import numpy as np
import torch

from isegm.inference import utils
from isegm.inference.clicker import Clicker


def inference(image, gt_mask, predictor, threshold=0.5, min_clicks=1, max_clicks=20):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []
    probs_list = []
    masks_list = []

    with torch.no_grad():
        predictor.set_input_image(image)

        for click_indx in range(max_clicks):
            clicker.make_next_click(pred_mask)
            pred_probs = predictor.get_prediction(clicker)
            pred_mask = pred_probs > threshold

            iou = utils.get_iou(gt_mask, pred_mask)
            ious_list.append(iou)

            probs_list.append(pred_probs.copy())
            masks_list.append(pred_mask.copy())

        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), probs_list, masks_list


def visualization(sample,
                  predictor,
                  mask=True,
                  score=False,
                  contour=True,
                  click=True,
                  threshold=0.5,
                  min_clicks=1,
                  max_clicks=20,
                  out_dir=None):
    mask = False if score else mask
    if out_dir is not None:
        out_dir = osp.join(out_dir, str(sample.sample_id))
        os.makedirs(out_dir, exist_ok=True)

    clicks, ious, probs, masks = inference(
        sample.image,
        sample.gt_mask(sample.objects_ids[0]),
        predictor,
        threshold=threshold,
        min_clicks=min_clicks,
        max_clicks=max_clicks)

    outputs = []

    show = cv2.cvtColor(sample.image.copy(), cv2.COLOR_RGB2BGR)
    gt_mask = sample.gt_mask(sample.objects_ids[0]).astype(np.bool8)
    if mask:
        show[~gt_mask] = (show[~gt_mask] * 0.5).astype(np.uint8)
    if contour:
        contours, _ = cv2.findContours(
            gt_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        show = cv2.drawContours(show, contours, -1, (106, 211, 253), 4)
    if out_dir is not None:
        cv2.imwrite(osp.join(out_dir, 'gt_mask.jpg'), show)
    outputs.append(show)

    for i in range(len(clicks)):
        show = cv2.cvtColor(sample.image.copy(), cv2.COLOR_RGB2BGR)

        if score:
            score_map = cv2.applyColorMap((probs[i] * 255).astype(np.uint8), cv2.COLORMAP_JET)
            show = cv2.addWeighted(show, 0.5, score_map, 0.5, 0)

        if mask:
            show[~masks[i]] = (show[~masks[i]] * 0.5).astype(np.uint8)

        if contour:
            contours, _ = cv2.findContours(masks[i].astype(np.uint8), cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_NONE)
            show = cv2.drawContours(show, contours, -1, (106, 211, 253), 4)

        if click:
            for j in range(i + 1):
                color = (80, 208, 146) if clicks[j].is_positive else (0, 0, 192)
                coords = (clicks[j].coords[1], clicks[j].coords[0])
                show = cv2.circle(show, coords, 11, (0, 0, 0), -1)
                show = cv2.circle(show, coords, 8, color, -1)

        outputs.append(show)

        if out_dir is not None:
            cv2.imwrite(osp.join(out_dir, f'{i+1}_{ious[i]:.2f}.jpg'), show)

    return outputs, ious
