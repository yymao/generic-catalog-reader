import warnings
from collections import defaultdict
import numpy as np
from .base import BaseGenericCatalog

__all__ = ['CompositeCatalog']


class CatalogWrapper(object):
    def __init__(self, instance, name, is_master=False, has_matching_format=False):
        self.instance = instance
        self.name = name
        self.is_master = bool(is_master)
        self.has_matching_format = bool(has_matching_format)
        self.iterator = None
        self.cache = None
        self.id_argsort = None


class CompositeCatalog(BaseGenericCatalog):

    def _subclass_init(
            self,
            catalog_instances,
            catalog_names=None,
            matching_column_name=None,
            allow_unsafe_match=False,
            **kwargs
    ):

        if catalog_names is None:
            catalog_names = ['catalog_{}'.format(i) for i in range(len(catalog_instances))]
        else:
            catalog_names = list(map(str, catalog_names))

        if len(set(catalog_names)) != len(catalog_names):
            raise ValueError('catalog names need to be all distinct!')

        self.catalogs = list()
        for name, instance in zip(map(str, catalog_names), catalog_instances):
            if name is None or instance is None:
                raise ValueError('catalog_instances and catalog_names need to have same length')
            self.catalogs.append(CatalogWrapper(instance, name))

        if len(self.catalogs) < 2:
            raise ValueError('need to have more than one catalogs to make a composite!')

        self.master_catalog = self.catalogs[0]
        for catalog in self.catalogs:
            if catalog.name == self.master_catalog.name:
                catalog.is_master = True
                catalog.has_matching_format = True
            if self.master_catalog.name not in getattr(catalog.instance, 'composite_compatible', []):
                if allow_unsafe_match:
                    warnings.warn('it\'s not safe to join these catalogs but I\'ll do it anyways')
                else:
                    raise ValueError('Not a valid match! {} it not an allowed master catalog for {}'.format(self.master_catalog.name, catalog.name))
            if self.master_catalog.name in getattr(catalog.instance, 'composite_matched_format', []):
                catalog.has_matching_format = True

        if all(catalog.has_matching_format for catalog in self.catalogs):
            self.matching_column_name = None
        else:
            self.matching_column_name = matching_column_name
            if not self.matching_column_name:
                raise ValueError('matching_column_name cannot be None or empty')

        self._native_filter_quantities = self.master_catalog.instance._native_filter_quantities
        self.native_filter_string_only = self.master_catalog.instance.native_filter_string_only

        self._quantity_modifiers = dict()
        for catalog in self.catalogs:
            for q in catalog.instance.list_all_quantities(True):
                self._quantity_modifiers[q] = (catalog.name, q)

    def _generate_native_quantity_list(self):
        return list(self._quantity_modifiers.values())

    def _obtain_native_data_dict(self, native_quantities_needed, native_quantity_getter):

        native_quantities_needed_dict = defaultdict(list)
        for name, q in native_quantities_needed:
            native_quantities_needed_dict[name].append(q)

        if self.matching_column_name:
            for catalog in self.catalogs:
                if catalog.is_master or not catalog.has_matching_format:
                    native_quantities_needed_dict[catalog.name].append(self.matching_column_name)

        data = dict()
        for catalog in self.catalogs:
            if catalog.name not in native_quantities_needed_dict:
                continue
            if catalog.has_matching_format:
                for q, v in catalog.instance._obtain_native_data_dict(
                    native_quantities_needed_dict[catalog.name],
                    native_quantity_getter[catalog.name]
                ).items():
                    data[(catalog.name, q)] = v
            elif catalog.cache is None:
                catalog.cache = catalog.instance.get_quantities(
                    native_quantities_needed_dict[catalog.name],
                )
                catalog.id_argsort = catalog.cache[self.matching_column_name].argsort()

        for catalog in self.catalogs:
            if catalog.name not in native_quantities_needed_dict or catalog.has_matching_format:
                continue
            s = np.searchsorted(
                a=catalog.cache[self.matching_column_name],
                v=data[(self.master_catalog.name, self.matching_column_name)],
                sorter=catalog.id_argsort,
            )
            matching_idx = catalog.id_argsort[s]
            matching_mask = catalog.cache[self.matching_column_name][matching_idx] == data[(self.master_catalog.name, self.matching_column_name)]

            for q in native_quantities_needed_dict[catalog.name]:
                data_this = catalog.cache[q][matching_idx]
                data_this[matching_mask] = np.nan #FIXME
                data[(catalog.name, q)] = data_this

        return data

    def _iter_native_dataset(self, native_filters=None):

        for catalog in self.catalogs:
            if catalog.has_matching_format:
                catalog.iterator = catalog.instance._iter_native_dataset(native_filters)
            else:
                catalog.cache = None

        for master_data in self.master_catalog.iterator:
            dataset = {self.master_catalog.name: master_data}
            for catalog in self.catalogs[1:]:
                if catalog.has_matching_format:
                    dataset[catalog.name] = next(catalog.iterator)
            yield dataset
