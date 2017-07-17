# generic-catalog-reader

We are still finalizing the API and schema for the GCR, so please expect possible future changes. (But it also means if you have suggestion regarding APT/schema, you can voice them!) 

Examples of using the GCR API can be found [here](https://github.com/LSSTDESC/generic-catalog-reader/tree/master/examples). (Thanks Joe for providing the CLF test!)

Currently there are two catalogs available:

1. "Proto-DC2" (AlphaQ) catalog by Eve Kovacs, Danila Korytov, Katrin Heitmann et al.
2. Buzzard v1.5 by Joe DeRose, Risa Wechsler et al.

The catalogs are served through NERSC. To use them, first clone thie repository on NERSC (yes, you need a NERSC account):

    git clone git@github.com:LSSTDESC/generic-catalog-reader.git 

then simply `cd` into `generic-catalog-reader/examples` to start an Jupyter notebook server.
