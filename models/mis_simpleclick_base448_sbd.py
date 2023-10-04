from isegm.utils.exp_imports.default import *
from mis.data import SBDUnsupervisedDataset
from mis.model.is_model import MISModel
from mis.model.losses import BCEWithLogitsLoss, SmoothLoss
from mis.training import MISTrainer


MODEL_NAME = 'sbd_plainvit_base448'


def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (448, 448)
    model_cfg.num_max_points = 24

    backbone_params = dict(
        img_size=model_cfg.crop_size,
        patch_size=(16, 16),
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
    )

    neck_params = dict(
        in_dim=768,
        out_dims=[128, 256, 512, 1024],
    )

    head_params = dict(
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        dropout_ratio=0.1,
        num_classes=1,
        loss_decode=None,
        align_corners=False,
        upsample=cfg.upsample,
        channels={
            'x1': 256,
            'x2': 128,
            'x4': 64
        }[cfg.upsample])

    model = MISModel(
        use_disks=True,
        norm_radius=5,
        with_prev_mask=True,
        backbone_params=backbone_params,
        neck_params=neck_params,
        head_params=head_params,
        random_split=cfg.random_split,
    )

    model.load_state_dict_from_url(cfg.IMAGENET_PRETRAINED_MODELS.MAE_BASE_URL)
    model.to(cfg.device)

    return model, model_cfg


def train(model, cfg, model_cfg):
    cfg.batch_size = 32 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_pseudo_loss = BCEWithLogitsLoss()
    loss_cfg.instance_pseudo_loss_weight = 1.0
    loss_cfg.instance_smooth_loss = SmoothLoss()
    loss_cfg.instance_smooth_loss_weight = 10.0

    train_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.25)),
        Flip(),
        RandomRotate90(),
        ShiftScaleRotate(
            shift_limit=0.03, scale_limit=0, rotate_limit=(-3, 3), border_mode=0, p=0.75),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size),
        RandomBrightnessContrast(
            brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
    ])

    points_sampler = MultiPointSampler(
        model_cfg.num_max_points,
        prob_gamma=0.80,
        merge_objects_prob=0.15,
        max_num_merged_objects=2)

    trainset = SBDUnsupervisedDataset(
        cfg.SBD_PATH,
        './data/proposals/sbd',
        split='train',
        augmentator=train_augmentator,
        min_object_area=80,
        keep_background_prob=0.01,
        points_sampler=points_sampler,
        decay=0.9)
    valset = trainset

    optimizer_params = {'lr': 5e-5, 'betas': (0.9, 0.999), 'eps': 1e-8}

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR, milestones=[50, 55], gamma=0.1)
    trainer = MISTrainer(
        model=model,
        cfg=cfg,
        model_cfg=model_cfg,
        loss_cfg=loss_cfg,
        trainset=trainset,
        valset=valset,
        optimizer='adam',
        optimizer_params=optimizer_params,
        layerwise_decay=cfg.layerwise_decay,
        lr_scheduler=lr_scheduler,
        checkpoint_interval=[(0, 20), (50, 1)],
        image_dump_interval=300,
        metrics=[AdaptiveIoU()],
        max_interactive_points=model_cfg.num_max_points,
        max_num_next_clicks=3)
    trainer.run(num_epochs=55, validation=False)
