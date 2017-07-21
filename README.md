# Generic Catalog Reader (GCR)

by [Yao-Yuan Mao](https://yymao.github.io)

**!! Warning !!** We are still finalizing the API and [schema](https://docs.google.com/document/d/1rUsImkBkjjw82Xa_-3a8VMV6K9aYJ8mXioaRhz0JoqI/edit#) for the GCR, so please expect possible future changes. (It also means that if you have suggestions regarding the API or schema, you can still voice them!) 

Currently there are two catalogs available:

1. "Proto-DC2" (AlphaQ) catalog by Eve Kovacs, Danila Korytov, Katrin Heitmann et al.
2. Buzzard v1.5 by Joe DeRose, Risa Wechsler et al.

We will add the specifications of these catalogs into the yaml config files that can be found [here](https://github.com/LSSTDESC/generic-catalog-reader/tree/master/catalogs).

Examples of using the GCR API can be found [here](https://github.com/LSSTDESC/generic-catalog-reader/tree/master/examples). (Thanks Joe for providing the CLF test!)

To use these catalogs with the GCR, first clone this repository on NERSC (yes, you need a NERSC account):

    git clone git@github.com:LSSTDESC/generic-catalog-reader.git 

And then, [start a NERSC notebook server](https://jupyter.nersc.gov) and browse to `generic-catalog-reader/examples` to start the example notebooks. You can copy these notebooks and then add your tests. 
