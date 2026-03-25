from .Resnet18 import resnet18
from .Resnet18_SD import resnet18_sd
from .Resnet_cifar import resnet18_cifar, resnet34_cifar, resnet50_cifar, resnet101_cifar, resnet152_cifar
from .Resnet_expert import resnet18_expert

METHOD_2_MODEL = {
    'er': resnet18,
    'scr': resnet18, 
    'joint': resnet18,
    'buf': resnet18,
    'mose': resnet18_sd,
    'dist': resnet18_cifar,
    'deep_expert': resnet18_expert
}

def get_model(method_name,  *args, **kwargs):
    if method_name in METHOD_2_MODEL.keys():
        return METHOD_2_MODEL[method_name](*args, **kwargs)
    else:
        raise Exception('unknown method!')