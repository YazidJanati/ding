from fm.samplers.pnp_flow import pnp_flow
from fm.samplers.reddiff import reddiff_sampler
from fm.samplers.flow_chef import flow_chef
from fm.samplers.flow_dps import flow_dps
from fm.samplers.psld import psld_sampler
from fm.samplers.daps import daps_sampler
from fm.samplers.resample import resample_sampler
from fm.samplers.exact_algo import ding
from fm.samplers.delayed_exact_algo import delayed_ding
from fm.samplers.alimama_controlnet import alimama_controlnet_sampler


AVAILABLE_SAMPLERS = {
    "pnp_flow": pnp_flow,
    "reddiff": reddiff_sampler,
    "flow_chef": flow_chef,
    "flow_dps": flow_dps,
    "psld": psld_sampler,
    "daps": daps_sampler,
    "resample": resample_sampler,
    "ding": ding,
    "delayed_ding": delayed_ding,
    "alimama_controlnet": alimama_controlnet_sampler,
}
