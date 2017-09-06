import yaml
from .BaseGalaxyCatalog import BaseGalaxyCatalog

_registered_readers = {}

def register_reader(subclass):
    """
    Registers a new galaxy catalog type with the loading utility.

    Parameters
    ----------
    subclass: subclass of BaseGalaxyCatalog
    """
    assert issubclass(subclass, BaseGalaxyCatalog) , "Provided class is not a subclass of BaseGalaxyCatalog"
    _registered_readers[subclass.__name__] = subclass

from .GalacticusGalaxyCatalog import GalacticusGalaxyCatalog
from .BuzzardGalaxyCatalog import BuzzardGalaxyCatalog
from .AlphaQGalaxyCatalog import AlphaQGalaxyCatalog

def _load_yaml_config(yaml_config_file):
    with open(yaml_config_file) as f:
        config = yaml.load(f.read())
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
    config = _load_yaml_config(yaml_config_file)
    if config_overwrite:
        config.update(config_overwrite)
    return _registered_readers[config['subclass_name']](**config)
