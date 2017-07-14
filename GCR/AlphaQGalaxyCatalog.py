"""
Alpha Q galaxy catalog class.
"""
from __future__ import division
import os
import numpy as np
import h5py
from astropy.cosmology import FlatLambdaCDM
from .BaseGalaxyCatalog import BaseGalaxyCatalog

__all__ = ['AlphaQGalaxyCatalog']

class AlphaQGalaxyCatalog(BaseGalaxyCatalog):
    """
    Alpha Q galaxy catalog class. Uses generic quantity and filter mechanisms
    defined by BaseGalaxyCatalog class.
    """

    def _subclass_init(self, filename, base_catalog_dir=os.curdir, **kwargs):

        self._quantity_modifiers = {
            'ra': None,
            'dec': None,
            'redshift_true': 'redshift',
            'shear_1': 'shear1',
            'shear_2': 'shear2',
            'halo_id': 'hostIndex',
            'halo_mass': 'hostHaloMass',
        }

        for band in 'ugriz':
            self._quantity_modifiers['mag_{}_any'.format(band)] = 'magnitude:SDSS_{}:observed'.format(band)
            self._quantity_modifiers['mag_{}_sdss'.format(band)] = 'magnitude:SDSS_{}:observed'.format(band)


        self._file = os.path.join(base_catalog_dir, filename)

        with h5py.File(self._file, 'r') as fh:
            self.cosmology = FlatLambdaCDM(
                H0=fh.attrs['H_0'],
                Om0=fh.attrs['Omega_matter'],
                Ob0=fh.attrs['Omega_b'],
            )


    def _generate_native_quantity_list(self):
        with h5py.File(self._file, 'r') as fh:
            native_quantities = set(fh.keys())
        return native_quantities


    def _iter_native_dataset(self, pre_filters=None):
        with h5py.File(self._file, 'r') as fh:
            yield fh


    @staticmethod
    def _fetch_native_quantity(dataset, native_quantity):
        return dataset[native_quantity].value