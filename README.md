# Generic Catalog Reader (GCR)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/gcr.svg)](https://anaconda.org/conda-forge/gcr)
[![PyPIVversion](https://img.shields.io/pypi/v/GCR.svg)](https://pypi.python.org/pypi/GCR)

A ready-to-use, abstract base class for creating a common reader interface that accesses generic table-like catalogs. 

This project was started in response to the need of the [DESCQA](https://github.com/LSSTDESC/descqa) framework. It is now used in the [LSST DESC GCR Catalogs](https://github.com/LSSTDESC/gcr-catalogs). 

## Installation

You can install `GCR` from conda-forge:

```bash
conda install gcr --channel conda-forge
```

Or from PyPI:

```bash
pip install GCR
```

## Concept

The reader should specify: (1) how to translate (assemble) requested quantities from the native quantities; and (2) how to access native quantities from the underlying data format. 

![Concept](https://i.imgur.com/eBR6kof.png)


## Usage

You can [find API documentation here](https://yymao.github.io/generic-catalog-reader/). However, looking at some [real examples](https://github.com/LSSTDESC/gcr-catalogs/tree/master/GCRCatalogs) is probably more useful. 

Basically, you will subclass `GCR.BaseGenericCatalog` and then set the member dict `_quantity_modifiers` inside `_subclass_init`, and implement the member methods `_generate_native_quantity_list` and `_iter_native_dataset`. Here's an minimal example. 

```python
import h5py
import GCR

class YourCatalogReader(GCR.BaseGenericCatalog):
    
    def _subclass_init(self, **kwargs):
        self._file = kwargs['filename']
        
        self._quantity_modifiers = {
            'galaxy_id' :    'galaxyID',
            'ra':            (lambda x: x/3600.0, 'ra'),
            'dec':           (lambda x: x/3600.0, 'dec'),
            'is_central':    (lambda x, y: x == y, 'haloId', 'parentHaloId'),
        }
        
    def _generate_native_quantity_list(self):
        """
        Must return an iterable of all native quantity names.
        """
        with h5py.File(self._file, 'r') as fh:
            return fh.keys()
        
    def _iter_native_dataset(self, native_filters=None):
        """
        Must be a generator.
        Must yield a callable, *native_quantity_getter*.
        This function must iterate over subsets of rows, not columns!
        Below are specifications of *native_quantity_getter*
        ---
        Must take a single argument of a native quantity name.
        Should assume the argument is valid.
        Must return a numpy 1d array.
        """
        assert not native_filters, '*native_filters* is not supported'
        with h5py.File(self._file, 'r') as fh:
            def native_quantity_getter(native_quantity):
                return fh[native_quantity].value
            yield native_quantity_getter
```
