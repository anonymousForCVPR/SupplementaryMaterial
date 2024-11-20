_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 8  # bs: total bs in all gpus
num_worker = batch_size
mix_prob = 0.8
empty_cache = False
enable_amp = True

max_test_aspect = 2
spa_native_mode = "warp"
weight = "exp/s3dis/semseg-pt-v3m1-0-rpe-T1/model/model_best.pth"

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="ModelWeightEMA", decay=0.999),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=13,
    backbone_out_channels=64,
    backbone=dict(
        type="SPPT",
        in_channels=6,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(128, 128, 128, 128, 128),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(128, 128, 128, 128),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=True,
        enable_flash=False,
        upcast_attention=True,
        upcast_softmax=True,
        enable_checkpoint=True,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        spa_cfg=dict(
            norm_loc="pre",
            pe_cfg=dict(
                num_layers=2,
                temperature=10,
            ),
            initializer_cfg=dict(
                init_cfg=dict(
                    type="square",
                    target_proxy_range=200,
                    proxy_search_range=[0.0, 1.0],
                    max_search_iter=10
                ),
                asso_cfg=dict(
                    type="fgrid",
                    dim=3,
                ),
                rel_bias_cfg=dict(
                    input_scale=2.0,
                    table_size=16,
                    temperature=[0.5, 2.5],     # random temperature
                    strength=1.0,
                ),
                reuse_proxy_feat=True,
                reuse_rel_bias=True,
            ),
            ca_cfg=dict(
                reduce_query="none",
                with_proj=True,
                same_kv=True,
                sim_scale=1,
            ),
            fuser_cfg=dict(
                mode="attn",
                with_ffn=True,
                rpe_cfg=dict(
                    input_scale=0.25,
                    table_size=16,
                    temperature=[0.5, 2.5],     # random temperature
                    strength=1.0,
                )
            ),
        ),
        spa_skip_layer=[1, 3],
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass",
             loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
epoch = 1000
optimizer = dict(type="AdamW", lr=0.0002, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.0005, 0.0005],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [
    dict(keyword="block", lr=0.0002),
]

# dataset settings
dataset_type = "S3DISDataset"
data_root = "data/s3dis"

data = dict(
    num_classes=13,
    ignore_index=-1,
    names=[
        "ceiling",
        "floor",
        "wall",
        "beam",
        "column",
        "window",
        "door",
        "table",
        "chair",
        "sofa",
        "bookcase",
        "board",
        "clutter",
    ],
    train=dict(
        type=dataset_type,
        split=("Area_2", "Area_3", "Area_4", "Area_5", "Area_6"),
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            ),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1],
                 axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate",
                 angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate",
                 angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=204800, mode="random", as_cube=True),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            # dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="Area_1",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="Copy",
                keys_dict={"coord": "origin_coord",
                           "segment": "origin_segment"},
            ),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "origin_coord",
                    "segment",
                    "origin_segment",
                ),
                offset_keys_dict=dict(
                    offset="coord", origin_offset="origin_coord"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="Area_1",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color", "normal"),
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("color", "normal"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
                [
                    dict(type="RandomScale", scale=[0.9, 0.9]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1, 1]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.1, 1.1]),
                    dict(type="RandomFlip", p=1),
                ],
            ],
        ),
    ),
)
