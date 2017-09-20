import yaml
import requests
from .base import BaseGalaxyCatalog

__all__ = ['register_reader', 'load_yaml', 'load_catalog']

_registered_readers = dict()

def register_reader(subclass):
    """
    Registers a new galaxy catalog type with the loading utility.

    Parameters
    ----------
    subclass: subclass of BaseGalaxyCatalog
    """
    assert issubclass(subclass, BaseGalaxyCatalog), "Provided class is not a subclass of BaseGalaxyCatalog"
    _registered_readers[subclass.__name__] = subclass


def load_yaml(yaml_file):
    """
    Load yaml file
    """
    try:
        r = requests.get(yaml_file, stream=True)
    except requests.exceptions.MissingSchema:
        with open(yaml_file) as f:
            config = yaml.load(f)
    else:
        r.raw.decode_content = True
        config = yaml.load(r.raw)
    return config


def load_catalog(yaml_config_file, config_overwrite=None):
    """
    Load a galaxy catalog as specified in a yaml config file.

    Parameters
    ----------
    yaml_config_file : str
        path to the yaml config file
    config_overwrite : dict, optional
        a dictionary of config options to overwrite

    Return
    ------
    galaxy_catalog : subclass of BaseGalaxyCatalog
    """
    config = load_yaml(yaml_config_file)
    if config_overwrite:
        config.update(config_overwrite)
    return _registered_readers[config['subclass_name']](**config)
