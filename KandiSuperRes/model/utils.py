from torch.nn import Identity


def exist(item):
    return item is not None


def set_default_item(condition, item_1, item_2=None):
    if condition:
        return item_1
    else:
        return item_2


def set_default_layer(condition, layer_1, args_1=[], kwargs_1={}, layer_2=Identity, args_2=[], kwargs_2={}):
    if condition:
        return layer_1(*args_1, **kwargs_1)
    else:
        return layer_2(*args_2, **kwargs_2)
