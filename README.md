# generic-catalog-reader

We are still finalizing the API and schema for the GCR, so please expect possible future changes. (But it also means if you have suggestion regarding APT/schema, you can voice them!) 

Here's an example of the most up-to-date API: 
    https://github.com/LSSTDESC/generic-catalog-reader/blob/master/examples/basic%20usage.ipynb 

To use it, just do 

    git clone git@github.com:LSSTDESC/generic-catalog-reader.git 

at NERSC edison, and then you can change into the example directory to start an Jupyter notebook server. 

The catalogs are served through NERSC. Though there is an `return_hdf5` option in the `get_quantities` function which you can just to generate a HDF5 file for download. 
