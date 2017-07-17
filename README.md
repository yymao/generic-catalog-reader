# Generic Catalog Reader (GCR)

by [Yao-Yuan Mao](https://yymao.github.io)

**Warming!** We are still finalizing the API and [schema](https://docs.google.com/document/d/1rUsImkBkjjw82Xa_-3a8VMV6K9aYJ8mXioaRhz0JoqI/edit#) for the GCR, so please expect possible future changes. (But it also means that if you have suggestions regarding the API or schema, you can still voice them!) 

Examples of using the GCR API can be found [here](https://github.com/LSSTDESC/generic-catalog-reader/tree/master/examples). (Thanks Joe for providing the CLF test!)

Currently there are two catalogs available:

1. "Proto-DC2" (AlphaQ) catalog by Eve Kovacs, Danila Korytov, Katrin Heitmann et al.
2. Buzzard v1.5 by Joe DeRose, Risa Wechsler et al.

The catalogs are served through NERSC. To use them, first clone thie repository on NERSC (yes, you need a NERSC account):

    git clone git@github.com:LSSTDESC/generic-catalog-reader.git 

then simply `cd` into `generic-catalog-reader/examples` to start an Jupyter notebook server.
