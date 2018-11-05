"""
Utility module
"""
import numpy as np
from numpy.core.records import fromarrays

__all__ = ['is_string_like', 'trivial_callable', 'dict_to_numpy_array', 'concatenate_1d']

def is_string_like(obj):
    """
    Check whether obj behaves like a string.
    """
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True

def trivial_callable(x):
    """return itself"""
    return x

def dict_to_numpy_array(d):
    """
    Convert a dict of 1d array to a numpy recarray
    """
    return fromarrays(d.values(), np.dtype([(str(k), v.dtype) for k, v in d.items()]))

def concatenate_1d(arrays):
    if len(arrays) == 0:
        return np.array([])
    if len(arrays) == 1:
        return np.asarray(arrays[0])
    return np.concatenate(arrays)
