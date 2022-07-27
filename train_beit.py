import os
import sys
from absl import app
from absl import flags
from absl import logging
import torch

from ml_collections import config_dict, config_flags

# Specify the default config settings
def get_config():
    config = config_dict.ConfigDict()
    config.model = config_dict.ConfigDict(
        dict(
            use_predefined_model = False,
            predefined_model = config_dict.placeholder(str),
            model_kwargs = dict(
                img_size=32,
                patch_size=2,
                in_chan=1,
                vocab_size=8192,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                drop_rate=0.3,
                attn_drop_rate=0.,
                drop_path_rate=0.,
                pre_train=True,
                use_abs_pos_emb=False,
                use_rel_pos_bias=True,
                use_relative_shared_pos_bias=True,
                head_init_scale=1e-2
            )
        )
    )
    config.train = config_dict.ConfigDict(
        dict(
            local_batch_size = 16,
            local_world_size = 1,
            local_rank = 0,
            num_gpus = torch.cuda.device_count()
        )
    )
    config.data = config_dict.ConfigDict(
        dict(
        )
    )
    config.optimizer = config_dict.ConfigDict(
        dict(
        )
    )

_CONFIG = config_flags.DEFINE_config_dict('config', get_config())

def main():
    pass

if __name__ == '__main__':
    flags.mark_flags_as_required('config')
    app.run(main)