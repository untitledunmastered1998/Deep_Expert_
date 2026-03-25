from .er import *
from .scr import *
from .mose import *
from .joint import *
from .buf import *
from .dist import *
from .dexpert import *

METHODS = {
    'er': ER,
    'scr': SCR,
    'mose': MOSE,
    'joint': Joint,
    'buf': Buf,
    'dist': Dist,
    'deep_expert': Dexpert,
    'dexpert': Dexpert,
}


def get_agent(method_name, *args, **kwargs):
    if method_name in METHODS.keys():
        return METHODS[method_name](*args, **kwargs)
    else:
        raise Exception('unknown method!')
