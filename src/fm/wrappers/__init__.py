from fm.wrappers.base import NetCFM
from fm.wrappers.sd3 import SD3Wrapper, SD3LatentWrapper


AVAILABLE_MODELS = {
    "sd3": SD3Wrapper,
    "sd3_latent": SD3LatentWrapper,
}
