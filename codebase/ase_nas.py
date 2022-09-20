import os
from codebase.third_party.ofa import OFAMobileNetV3
from codebase.third_party.ofa.representation import OFAArchitecture

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

WIDTH = 1.2
archs_strings = {
    'ase_nas': '3,3,3,4,4:5,5,7,0,5,5,7,0,5,7,5,0,5,5,5,7,5,7,7,7:6,6,3,0,6,4,6,0,6,6,6,0,6,6,6,4,6,6,4,4',
    'ase_nas_plus': '4,4,3,4,4:3,5,5,5,7,7,3,7,5,3,7,0,5,5,7,7,3,5,5,3:6,6,3,3,3,6,3,6,6,3,6,0,6,3,6,4,6,6,4,6',
}


archs_weights_urls = {
    "ase_nas": "https://github.com/chenyaofo/ASE-NAS/releases/download/weights/ase-nas-ba5edf7c.pt",
    "ase_nas_plus": "https://github.com/chenyaofo/ASE-NAS/releases/download/weights/ase-nas-plus-7cad288d.pt",
}

# download with proxy for China mainland users
if os.environ.get("CN", None) == "true":
    for k, v in archs_weights_urls.items():
        archs_weights_urls[k] = "https://ghproxy.com/" + v


def _get_pnag_from_supernet(name):
    ofa_supernet = OFAMobileNetV3(
        dropout_rate=0.1,
        width_mult_list=WIDTH,
        ks_list=[3, 5, 7],
        expand_ratio_list=[3, 4, 6],
        depth_list=[2, 3, 4],
    )
    arch = OFAArchitecture.from_legency_string(archs_strings[name])
    ofa_supernet.set_active_subnet(ks=arch.ks, e=arch.ratios, d=arch.depths)
    subnet = ofa_supernet.get_active_subnet(preserve_weight=False)

    state_dict = load_state_dict_from_url(archs_weights_urls[name], progress=True, map_location="cpu")
    subnet.load_state_dict(state_dict)

    return subnet


def ase_nas():
    return _get_pnag_from_supernet("ase_nas")


def ase_nas_plus():
    return _get_pnag_from_supernet("ase_nas_plus")

