import random

import numpy as np
import torch

from isegm.engine.trainer import ISTrainer, get_next_points


class MISTrainer(ISTrainer):

    def batch_forward(self, batch_data, validation=False):
        metrics = self.val_metrics if validation else self.train_metrics
        losses_logging = dict()

        with torch.set_grad_enabled(not validation):
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            image, gt_mask, points = batch_data['images'], batch_data['instances'], batch_data[
                'points']
            orig_image, orig_gt_mask, _ = image.clone(), gt_mask.clone(), points.clone()

            prev_output = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]

            last_click_indx = None

            with torch.no_grad():
                num_iters = random.randint(0, self.max_num_next_clicks)

                for click_indx in range(num_iters):
                    last_click_indx = click_indx

                    if not validation:
                        self.net.eval()

                    if self.click_models is None or click_indx >= len(self.click_models):
                        eval_model = self.net
                    else:
                        eval_model = self.click_models[click_indx]

                    net_input = torch.cat(
                        (image, prev_output), dim=1) if self.net.with_prev_mask else image
                    prev_output = torch.sigmoid(eval_model(net_input, points)['instances'])

                    points = get_next_points(prev_output, orig_gt_mask, points, click_indx + 1)

                    if not validation:
                        self.net.train()

                if self.net.with_prev_mask and self.prev_mask_drop_prob > 0 and last_click_indx is not None:
                    zero_mask = np.random.random(
                        size=prev_output.size(0)) < self.prev_mask_drop_prob
                    prev_output[zero_mask] = torch.zeros_like(prev_output[zero_mask])

            batch_data['points'] = points

            net_input = torch.cat((image, prev_output), dim=1) if self.net.with_prev_mask else image
            output = self.net(net_input, points)

            loss = 0.0
            for loss_name in self.loss_cfg.keys():
                if loss_name.endswith('weight'):
                    continue
                if loss_name.startswith('instance'):
                    pred = output['instances']
                elif loss_name.startswith('instance_aux'):
                    pred = output['instances_aux']
                else:
                    continue
                loss = self.add_loss(loss_name, loss, losses_logging, validation,
                                     lambda: [pred, orig_gt_mask, orig_image * 255])

            if self.is_master:
                with torch.no_grad():
                    for m in metrics:
                        m.update(*(output.get(x) for x in m.pred_outputs),
                                 *(batch_data[x] for x in m.gt_outputs))

        return loss, losses_logging, batch_data, output

    def validation(self, epoch):
        pass
