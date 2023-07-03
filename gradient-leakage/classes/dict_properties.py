class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def dict_properties(d):
    if isinstance(d, dict):
        dotdict = DotDict({key: dict_properties(value) for key, value in d.items()})
        return dotdict
    else:
        return d