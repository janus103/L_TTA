import os
from typing import Any, Dict, Optional, Union
from urllib.parse import urlsplit

from .pretrained import PretrainedCfg, split_model_name_tag
from .helpers import load_checkpoint
from .hub import load_model_config_from_hf
from .layers import set_layer_config
from .registry import is_model, model_entrypoint


def parse_model_name(model_name):
    if model_name.startswith('hf_hub'):
        # NOTE for backwards compat, deprecate hf_hub use
        model_name = model_name.replace('hf_hub', 'hf-hub')
    parsed = urlsplit(model_name)
    assert parsed.scheme in ('', 'timm', 'hf-hub')
    if parsed.scheme == 'hf-hub':
        # FIXME may use fragment as revision, currently `@` in URI path
        return parsed.scheme, parsed.path
    else:
        model_name = os.path.split(parsed.path)[-1]
        return 'timm', model_name


def safe_model_name(model_name, remove_source=True):
    # return a filename / path safe model name
    def make_safe(name):
        return ''.join(c if c.isalnum() else '_' for c in name).rstrip('_')
    if remove_source:
        model_name = parse_model_name(model_name)[-1]
    return make_safe(model_name)


def create_model(
        model_name: str,
        pretrained: bool = False,
        pretrained_cfg: Optional[Union[str, Dict[str, Any], PretrainedCfg]] = None,
        pretrained_cfg_overlay:  Optional[Dict[str, Any]] = None,
        checkpoint_path: str = '',
        scriptable: Optional[bool] = None,
        exportable: Optional[bool] = None,
        no_jit: Optional[bool] = None,
        aux_header = False,
        no_skip = False,
        dwt_kernel_size = [0, 0, 0],
        dwt_level = [2, 2, 2],
        dwt_bn = [0, 0, 0],
        deep_format = False,
        mvar=False,
        meta_option=0,
        **kwargs,
):
    """Create a model

    Lookup model's entrypoint function and pass relevant args to create a new model.

    **kwargs will be passed through entrypoint fn to timm.models.build_model_with_cfg()
    and then the model class __init__(). kwargs values set to None are pruned before passing.

    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        pretrained_cfg (Union[str, dict, PretrainedCfg]): pass in external pretrained_cfg for model
        pretrained_cfg_overlay (dict): replace key-values in base pretrained_cfg with these
        checkpoint_path (str): path of checkpoint to load _after_ the model is initialized
        scriptable (bool): set layer config so that model is jit scriptable (not working for all models yet)
        exportable (bool): set layer config so that model is traceable / ONNX exportable (not fully impl/obeyed yet)
        no_jit (bool): set layer config so that model doesn't utilize jit scripted layers (so far activations only)

    Keyword Args:
        drop_rate (float): dropout rate for training (default: 0.0)
        global_pool (str): global pool type (default: 'avg')
        **: other kwargs are consumed by builder or model __init__()
    """
    # Parameters that aren't supported by all models or are intended to only override model defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting models don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    model_source, model_name = parse_model_name(model_name)
    if model_source == 'hf-hub':
        assert not pretrained_cfg, 'pretrained_cfg should not be set when sourcing model from Hugging Face Hub.'
        # For model names specified in the form `hf-hub:path/architecture_name@revision`,
        # load model weights + pretrained_cfg from Hugging Face hub.
        pretrained_cfg, model_name = load_model_config_from_hf(model_name)
    else:
        model_name, pretrained_tag = split_model_name_tag(model_name)
        if not pretrained_cfg:
            # a valid pretrained_cfg argument takes priority over tag in model name
            pretrained_cfg = pretrained_tag

    if not is_model(model_name):
        raise RuntimeError('Unknown model (%s)' % model_name)
    print('Model Name = {}'.format(model_name))
    create_fn = model_entrypoint(model_name)
    #print('create_fn @@@@@@@@@', create_fn)
    if model_name == 'densenet121' or model_name == 'vgg16':
        with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
            model = create_fn(
                pretrained=pretrained,
                pretrained_cfg=pretrained_cfg,
                pretrained_cfg_overlay=pretrained_cfg_overlay,
                aux_header=aux_header,
                #no_skip=no_skip,
                #dwt_kernel_size=dwt_kernel_size,
                #dwt_level=dwt_level,
                #dwt_bn=dwt_bn,
                #deep_format=deep_format,
                **kwargs,
            )
    elif model_name == 'deit_tiny_patch16_224':
        with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
            model = create_fn(
                pretrained=pretrained,
                pretrained_cfg=pretrained_cfg,
                pretrained_cfg_overlay=pretrained_cfg_overlay,
                dwt_level=dwt_level,
                **kwargs,
            )
    else:
        # print('LOG_CHECKER:: SWIN Transformer Here = YES!!!')
        with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
            model = create_fn(
                pretrained=pretrained,
                pretrained_cfg=pretrained_cfg,
                pretrained_cfg_overlay=pretrained_cfg_overlay,
                aux_header=aux_header,
                no_skip=no_skip,
                dwt_kernel_size=dwt_kernel_size,
                dwt_level=dwt_level,
                dwt_bn=dwt_bn,
                deep_format=deep_format,
                mvar=mvar,
                meta_option=meta_option,
                **kwargs,
            )

    # print('@@@@@@@@@@@@@@@@@@@@ checkpoint_pth {}'.format(checkpoint_path))

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)

    return model
