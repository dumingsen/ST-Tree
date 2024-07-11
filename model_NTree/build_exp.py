# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .swin_mlp import SwinMLP
# from .swin_tree import ViT_NeT
from .swin_tree_exp import ViT_NeT


def build_model(config,
                t=1,  # 长度
                down_dim=1024,  # length = 1536 * 2，降维维度
                hidden_dim=(96, 62),  ##192
                layers1=(2, 2, 6, 2),
                heads=(3, 6, 12, 24),
                channels=1,
                num_classes=1,
                head_dim=32,
                window_size=1,
                downscaling_factors=(4, 2, 2, 2),  # 代表多长的时间作为一个特征
                relative_pos_embedding=True,
                wa=1,
                prob=1,
                mask=1, ):
    model_type = config.MODEL.TYPE


    if model_type == 'swin_tree':
        model = ViT_NeT(config,
                        t=t,  # 长度
                        down_dim=down_dim,  # length = 1536 * 2，降维维度
                        hidden_dim=hidden_dim,
                        layers=layers1,
                        heads=heads,
                        channels=channels,
                        num_classes=num_classes,
                        head_dim=head_dim,
                        window_size=window_size,
                        downscaling_factors=downscaling_factors,  # 代表多长的时间作为一个特征
                        wa=wa,
                        prob=prob,
                        mask=mask, )  ###模型
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
