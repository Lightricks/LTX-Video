def make_hashable_key(dict_key):
    def convert_value(value):
        if isinstance(value, list):
            return tuple(value)
        elif isinstance(value, dict):
            return tuple(sorted((k, convert_value(v)) for k, v in value.items()))
        else:
            return value

    return tuple(sorted((k, convert_value(v)) for k, v in dict_key.items()))


DIFFUSERS_SCHEDULER_CONFIG = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "_diffusers_version": "0.32.0.dev0",
    "base_image_seq_len": 1024,
    "base_shift": 0.95,
    "invert_sigmas": False,
    "max_image_seq_len": 4096,
    "max_shift": 2.05,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": 0.1,
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
DIFFUSERS_TRANSFORMER_CONFIG = {
    "_class_name": "LTXVideoTransformer3DModel",
    "_diffusers_version": "0.32.0.dev0",
    "activation_fn": "gelu-approximate",
    "attention_bias": True,
    "attention_head_dim": 64,
    "attention_out_bias": True,
    "caption_channels": 4096,
    "cross_attention_dim": 2048,
    "in_channels": 128,
    "norm_elementwise_affine": False,
    "norm_eps": 1e-06,
    "num_attention_heads": 32,
    "num_layers": 28,
    "out_channels": 128,
    "patch_size": 1,
    "patch_size_t": 1,
    "qk_norm": "rms_norm_across_heads",
}
DIFFUSERS_VAE_CONFIG = {
    "_class_name": "AutoencoderKLLTXVideo",
    "_diffusers_version": "0.32.0.dev0",
    "block_out_channels": [128, 256, 512, 512],
    "decoder_causal": False,
    "encoder_causal": True,
    "in_channels": 3,
    "latent_channels": 128,
    "layers_per_block": [4, 3, 3, 3, 4],
    "out_channels": 3,
    "patch_size": 4,
    "patch_size_t": 1,
    "resnet_norm_eps": 1e-06,
    "scaling_factor": 1.0,
    "spatio_temporal_scaling": [True, True, True, False],
}

OURS_SCHEDULER_CONFIG = {
    "_class_name": "RectifiedFlowScheduler",
    "_diffusers_version": "0.25.1",
    "num_train_timesteps": 1000,
    "shifting": "SD3",
    "base_resolution": None,
    "target_shift_terminal": 0.1,
}

OURS_TRANSFORMER_CONFIG = {
    "_class_name": "Transformer3DModel",
    "_diffusers_version": "0.25.1",
    "_name_or_path": "PixArt-alpha/PixArt-XL-2-256x256",
    "activation_fn": "gelu-approximate",
    "attention_bias": True,
    "attention_head_dim": 64,
    "attention_type": "default",
    "caption_channels": 4096,
    "cross_attention_dim": 2048,
    "double_self_attention": False,
    "dropout": 0.0,
    "in_channels": 128,
    "norm_elementwise_affine": False,
    "norm_eps": 1e-06,
    "norm_num_groups": 32,
    "num_attention_heads": 32,
    "num_embeds_ada_norm": 1000,
    "num_layers": 28,
    "num_vector_embeds": None,
    "only_cross_attention": False,
    "out_channels": 128,
    "project_to_2d_pos": True,
    "upcast_attention": False,
    "use_linear_projection": False,
    "qk_norm": "rms_norm",
    "standardization_norm": "rms_norm",
    "positional_embedding_type": "rope",
    "positional_embedding_theta": 10000.0,
    "positional_embedding_max_pos": [20, 2048, 2048],
    "timestep_scale_multiplier": 1000,
}
OURS_VAE_CONFIG = {
    "_class_name": "CausalVideoAutoencoder",
    "dims": 3,
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 128,
    "blocks": [
        ["res_x", 4],
        ["compress_all", 1],
        ["res_x_y", 1],
        ["res_x", 3],
        ["compress_all", 1],
        ["res_x_y", 1],
        ["res_x", 3],
        ["compress_all", 1],
        ["res_x", 3],
        ["res_x", 4],
    ],
    "scaling_factor": 1.0,
    "norm_layer": "pixel_norm",
    "patch_size": 4,
    "latent_log_var": "uniform",
    "use_quant_conv": False,
    "causal_decoder": False,
}


diffusers_and_ours_config_mapping = {
    make_hashable_key(DIFFUSERS_SCHEDULER_CONFIG): OURS_SCHEDULER_CONFIG,
    make_hashable_key(DIFFUSERS_TRANSFORMER_CONFIG): OURS_TRANSFORMER_CONFIG,
    make_hashable_key(DIFFUSERS_VAE_CONFIG): OURS_VAE_CONFIG,
}


TRANSFORMER_KEYS_RENAME_DICT = {
    "proj_in": "patchify_proj",
    "time_embed": "adaln_single",
    "norm_q": "q_norm",
    "norm_k": "k_norm",
}


VAE_KEYS_RENAME_DICT = {
    "decoder.up_blocks.3.conv_in": "decoder.up_blocks.7",
    "decoder.up_blocks.3.upsamplers.0": "decoder.up_blocks.8",
    "decoder.up_blocks.3": "decoder.up_blocks.9",
    "decoder.up_blocks.2.upsamplers.0": "decoder.up_blocks.5",
    "decoder.up_blocks.2.conv_in": "decoder.up_blocks.4",
    "decoder.up_blocks.2": "decoder.up_blocks.6",
    "decoder.up_blocks.1.upsamplers.0": "decoder.up_blocks.2",
    "decoder.up_blocks.1": "decoder.up_blocks.3",
    "decoder.up_blocks.0": "decoder.up_blocks.1",
    "decoder.mid_block": "decoder.up_blocks.0",
    "encoder.down_blocks.3": "encoder.down_blocks.8",
    "encoder.down_blocks.2.downsamplers.0": "encoder.down_blocks.7",
    "encoder.down_blocks.2": "encoder.down_blocks.6",
    "encoder.down_blocks.1.downsamplers.0": "encoder.down_blocks.4",
    "encoder.down_blocks.1.conv_out": "encoder.down_blocks.5",
    "encoder.down_blocks.1": "encoder.down_blocks.3",
    "encoder.down_blocks.0.conv_out": "encoder.down_blocks.2",
    "encoder.down_blocks.0.downsamplers.0": "encoder.down_blocks.1",
    "encoder.down_blocks.0": "encoder.down_blocks.0",
    "encoder.mid_block": "encoder.down_blocks.9",
    "conv_shortcut.conv": "conv_shortcut",
    "resnets": "res_blocks",
    "norm3": "norm3.norm",
    "latents_mean": "per_channel_statistics.mean-of-means",
    "latents_std": "per_channel_statistics.std-of-means",
}
