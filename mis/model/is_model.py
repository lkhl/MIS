import torch
from torchvision import transforms

from isegm.model.is_plainvit_model import PlainVitModel
from isegm.model.modeling.pos_embed import interpolate_pos_embed


class MISModel(PlainVitModel):

    def __init__(self, **kwargs):
        super(MISModel, self).__init__(**kwargs)
        self.transform = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.3),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur((3, 3), (1.0, 2.0))], p=0.2)
        ])

    def augmentation(self, image):
        prev_mask = None
        if self.with_prev_mask:
            prev_mask = image[:, 3:, :, :]
            image = image[:, :3, :, :]
        image = torch.stack([self.transform(x) for x in image], dim=0)
        if prev_mask is not None:
            image = torch.cat((image, prev_mask), dim=1)
        return image

    def forward(self, image, points):
        if self.training:
            image = self.augmentation(image)
        return super(MISModel, self).forward(image, points)

    def load_state_dict_from_url(self, url):
        checkpoint = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        print('Load pre-trained checkpoint from: %s' % url)
        checkpoint_model = checkpoint['model']

        # interpolate position embedding
        interpolate_pos_embed(self.backbone, checkpoint_model)

        # load pre-trained model
        msg = self.backbone.load_state_dict(checkpoint_model, strict=False)
        print(msg)
