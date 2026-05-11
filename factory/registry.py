"""Model and trainer registries with decorator-based registration.

Add new models by decorating a builder function:
    @register_model('my_new_arch')
    def build_my_model(cfg, device):
        ...
        return model
"""

MODEL_REGISTRY = {}
TRAINER_REGISTRY = {}


def register_model(name):
    """Decorator to register a model builder function."""
    def decorator(fn):
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator


def register_trainer(name):
    """Decorator to register a trainer class."""
    def decorator(cls):
        TRAINER_REGISTRY[name] = cls
        return cls
    return decorator


# ============================================================
# Built-in model builders
# ============================================================

@register_model('direct_unet')
def build_direct_unet(cfg, device):
    from model.direct_unet import DirectUNet
    model = DirectUNet(
        cond_ch=cfg.model.cond_ch,
        base_ch=cfg.model.base_ch,
        latent_ch=cfg.model.latent_ch,
        use_polar_moe=cfg.model.use_polar_moe,
        use_circular_conv=cfg.model.use_circular_conv,
        use_coord_embed=cfg.model.use_coord_embed,
        use_spherical_attn=cfg.model.use_spherical_attn,
        hr_size=tuple(cfg.data.hr_size),
        ms_injection=cfg.model.ms_injection,
        out_ch=cfg.model.out_ch,
    ).to(device)
    return model


@register_model('center_growing')
def build_center_growing(cfg, device):
    from model.center_growing import CenterGrowingUNet
    model = CenterGrowingUNet(
        in_ch=cfg.model.get('in_ch', 10),
        out_ch=cfg.model.get('out_ch', 3),
        base_ch=cfg.model.get('base_ch', 64),
    ).to(device)
    return model


@register_model('multiscale_consistency')
def build_multiscale_consistency(cfg, device):
    from model.multiscale_consistency import MultiScaleConsistencyUNet
    model = MultiScaleConsistencyUNet(
        in_ch=cfg.model.get('in_ch', 3),
        out_ch=cfg.model.get('out_ch', 3),
        base_ch=cfg.model.get('base_ch', 64),
    ).to(device)
    return model


@register_model('iterative_refinement')
def build_iterative_refinement(cfg, device):
    from model.iterative_refinement import IterativeRefinementUNet
    model = IterativeRefinementUNet(
        in_ch=cfg.model.get('in_ch', 7),
        out_ch=cfg.model.get('out_ch', 3),
        base_ch=cfg.model.get('base_ch', 64),
    ).to(device)
    return model


@register_model('multiscale_center_mask')
def build_multiscale_center_mask(cfg, device):
    from model.multiscale_center_mask import MultiScaleCenterMaskUNet
    model = MultiScaleCenterMaskUNet(
        in_ch=cfg.model.get('in_ch', 3),
        out_ch=cfg.model.get('out_ch', 3),
        base_ch=cfg.model.get('base_ch', 64),
    ).to(device)
    return model


@register_model('multiscale_film')
def build_multiscale_film(cfg, device):
    from model.multiscale_film import MultiScaleFiLMUNet
    model = MultiScaleFiLMUNet(
        in_ch=cfg.model.get('in_ch', 3),
        out_ch=cfg.model.get('out_ch', 3),
        base_ch=cfg.model.get('base_ch', 64),
    ).to(device)
    return model


@register_model('multiscale_coord')
def build_multiscale_coord(cfg, device):
    from model.multiscale_coord import MultiScaleCoordUNet
    model = MultiScaleCoordUNet(
        in_ch=cfg.model.get('in_ch', 3),
        out_ch=cfg.model.get('out_ch', 3),
        base_ch=cfg.model.get('base_ch', 64),
    ).to(device)
    return model


@register_model('focus_refine')
def build_focus_refine(cfg, device):
    from model.focus_refine import FocusRefineUNet
    model = FocusRefineUNet(
        in_ch=cfg.model.get('in_ch', 4),
        base_ch=cfg.model.get('base_ch', 64),
        out_ch=cfg.model.get('out_ch', 3),
        time_dim=cfg.model.get('time_dim', 128),
        use_spatial_bias=cfg.model.get('use_spatial_bias', True),
        use_time_embedding=cfg.model.get('use_time_embedding', True),
        hr_size=tuple(cfg.data.hr_size),
    ).to(device)
    return model


@register_model('ssm_focus')
def build_ssm_focus(cfg, device):
    from model.ssm_focus import SSMFocusUNet
    model = SSMFocusUNet(
        in_ch=cfg.model.get('in_ch', 7),
        out_ch=cfg.model.get('out_ch', 3),
        base_ch=cfg.model.get('base_ch', 64),
        time_dim=cfg.model.get('time_dim', 128),
        d_state=cfg.model.get('d_state', 16),
        d_conv=cfg.model.get('d_conv', 4),
        expand=cfg.model.get('expand', 2),
        use_fixed_delta=cfg.model.get('use_fixed_delta', False),
        scan_mode=cfg.model.get('scan_mode', 'bidirectional'),
        use_mask=cfg.model.get('use_mask', True),
    ).to(device)
    return model


@register_model('direct_sr')
def build_direct_sr(cfg, device):
    from model.focus_refine import DirectSRUNet
    model = DirectSRUNet(
        in_ch=cfg.model.get('in_ch', 3),
        base_ch=cfg.model.get('base_ch', 64),
        out_ch=cfg.model.get('out_ch', 3),
    ).to(device)
    return model


@register_model('ib_focus')
def build_ib_focus(cfg, device):
    from model.ib_focus import IBFocusUNet
    model = IBFocusUNet(
        in_ch=cfg.model.get('in_ch', 4),
        base_ch=cfg.model.get('base_ch', 64),
        out_ch=cfg.model.get('out_ch', 3),
        time_dim=cfg.model.get('time_dim', 128),
        hr_size=tuple(cfg.data.hr_size),
        sigma_max=cfg.model.get('sigma_max', 0.3),
        sigma_min=cfg.model.get('sigma_min', 0.05),
    ).to(device)
    return model


@register_model('dit_multistep')
def build_dit_multistep(cfg, device):
    from model.dit_focus import MultiStepDiT
    model = MultiStepDiT(
        in_ch=cfg.model.get('in_ch', 7),
        dim=cfg.model.get('dim', 512),
        depth=cfg.model.get('depth', 8),
        num_heads=cfg.model.get('num_heads', 8),
        patch_size=cfg.model.get('patch_size', 16),
        mlp_ratio=cfg.model.get('mlp_ratio', 4),
    ).to(device)
    return model


@register_model('dit_singlestep')
def build_dit_singlestep(cfg, device):
    from model.dit_focus import SingleStepDiT
    # Convert list format → dict: [[2,c64,64,64],[4,c128,128,128],[7,full,null,null]]
    raw = cfg.model.get('head_specs', [[2, 'c64', 64, 64], [4, 'c128', 128, 128], [6, 'full', None, None]])
    head_specs = {}
    for item in raw:
        layer_idx, name, rh, rw = item[0], item[1], item[2], item[3]
        if rw is None or rh is None:
            head_specs[int(layer_idx)] = (str(name), None, None)
        else:
            head_specs[int(layer_idx)] = (str(name), int(rh), int(rw))
    model = SingleStepDiT(
        in_ch=cfg.model.get('in_ch', 4),
        dim=cfg.model.get('dim', 512),
        depth=cfg.model.get('depth', 8),
        num_heads=cfg.model.get('num_heads', 8),
        patch_size=cfg.model.get('patch_size', 16),
        mlp_ratio=cfg.model.get('mlp_ratio', 4),
        head_specs=head_specs,
    ).to(device)
    return model


@register_model('wavelet_flow')
def build_wavelet_flow(cfg, device):
    from model.wavelet_flow import WaveletVFE
    model = WaveletVFE(
        base_ch=cfg.model.get('base_ch', 64),
        n_blocks=cfg.model.get('n_blocks', 8),
        time_dim=cfg.model.get('time_dim', 256),
    ).to(device)
    return model


@register_model('wavelet_direct')
def build_wavelet_direct(cfg, device):
    from model.wavelet_flow import WaveletDirectSR
    model = WaveletDirectSR(
        base_ch=cfg.model.get('base_ch', 64),
        n_blocks=cfg.model.get('n_blocks', 8),
    ).to(device)
    return model


@register_model('morpho_sr')
def build_morpho_sr(cfg, device):
    from model.morpho_sr import MorphoSR
    model = MorphoSR(
        state_ch=cfg.model.get('state_ch', 16),
        hidden_ch=cfg.model.get('hidden_ch', 64),
        cond_ch=cfg.model.get('cond_ch', 0),
        use_pre_encoder=cfg.model.get('use_pre_encoder', False),
        pre_encoder_ch=cfg.model.get('pre_encoder_ch', 32),
        pre_encoder_n=cfg.model.get('pre_encoder_n', 4),
        inject_cond=cfg.model.get('inject_cond', False),
        zero_init=cfg.model.get('zero_init', False),
    ).to(device)
    return model


# ============================================================
# Lazy register built-in trainers
# ============================================================

def _register_builtin_trainers():
    if 'direct' not in TRAINER_REGISTRY:
        from factory.trainer import DirectTrainer
        TRAINER_REGISTRY['direct'] = DirectTrainer
    if 'gan' not in TRAINER_REGISTRY:
        from factory.trainer_gan import GANTrainer
        TRAINER_REGISTRY['gan'] = GANTrainer
    if 'center_growing' not in TRAINER_REGISTRY:
        from factory.trainer_cg import CenterGrowingTrainer
        TRAINER_REGISTRY['center_growing'] = CenterGrowingTrainer
    if 'multiscale' not in TRAINER_REGISTRY:
        from factory.trainer_multiscale import MultiScaleTrainer
        TRAINER_REGISTRY['multiscale'] = MultiScaleTrainer
    if 'refinement' not in TRAINER_REGISTRY:
        from factory.trainer_refine import RefinementTrainer
        TRAINER_REGISTRY['refinement'] = RefinementTrainer
    if 'multiscale_center' not in TRAINER_REGISTRY:
        from factory.trainer_multiscale_v2 import MultiScaleCenterTrainer
        TRAINER_REGISTRY['multiscale_center'] = MultiScaleCenterTrainer
    if 'focus_refine' not in TRAINER_REGISTRY:
        from factory.trainer_focus_refine import FocusRefineTrainer
        TRAINER_REGISTRY['focus_refine'] = FocusRefineTrainer
    if 'ib_focus' not in TRAINER_REGISTRY:
        from factory.trainer_ib_focus import IBFocusTrainer
        TRAINER_REGISTRY['ib_focus'] = IBFocusTrainer
    if 'wavelet_flow' not in TRAINER_REGISTRY:
        from factory.trainer_wavelet_flow import WaveletFlowTrainer
        TRAINER_REGISTRY['wavelet_flow'] = WaveletFlowTrainer
    if 'wavelet_direct' not in TRAINER_REGISTRY:
        from factory.trainer_wavelet_flow import WaveletDirectTrainer
        TRAINER_REGISTRY['wavelet_direct'] = WaveletDirectTrainer
    if 'morpho' not in TRAINER_REGISTRY:
        from factory.trainer_morpho import MorphoTrainer
        TRAINER_REGISTRY['morpho'] = MorphoTrainer
