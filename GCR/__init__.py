from .BaseGalaxyCatalog import BaseGalaxyCatalog
from .GalacticusGalaxyCatalog import GalacticusGalaxyCatalog
from .BuzzardGalaxyCatalog import BuzzardGalaxyCatalog
from .AlphaQGalaxyCatalog import AlphaQGalaxyCatalog

import yaml

def load_yaml_config(filename):
    with open(filename) as f:
        config = yaml.load(f.read())
    return config