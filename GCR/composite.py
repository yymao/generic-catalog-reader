"""
Composite catalog reader
"""
import warnings
from collections import defaultdict
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest
import numpy as np
from .base import BaseGenericCatalog

__all__ = ['CompositeCatalog', 'MATCHING_FORMAT', 'MATCHING_ORDER']

# define module constants to be used in matching_columns
MATCHING_FORMAT = None
MATCHING_ORDER = tuple()


class CatalogWrapper(object):
    """
    A simple wrapper to enhance code readability
    """
    def __init__(self, instance, identifier, matching_method, is_master):
        self.instance = instance
        self.identifier = identifier
        self.is_master = bool(is_master)
        self.matching_method = matching_method
        self.matching_column = matching_method
        self.matching_format = (matching_method == MATCHING_FORMAT) or self.is_master
        self.matching_order = (matching_method == MATCHING_ORDER)
        if self.matching_format or self.matching_order:
            self.need_index_matching = False
        else:
            self.need_index_matching = True
        self.iterator = None
        self.cache = None
        self.sorter = None
        self.counter = None

    def clear(self):
        """
        clear cache
        """
        self.iterator = None
        self.cache = None
        self.counter = None


class CompositeCatalog(BaseGenericCatalog):
    """
    Composite Catalog

    Parameters
    ----------
    catalog_instances : list/tuple of GCR.BaseGenericCatalog instances
        The first element of `catalog_instances` is the master catalog,
        with the subsequent elements the "add-on" catalogs that will overwrite
        the columns of the master catlog.

    catalog_identifiers : list/tuple of string, or None
        A list/tuple of identifiers that correspond to `catalog_instances`.
        Each identifier is usually a string.
        Should have the same length as `catalog_instances`.
        If set to `None`, the identifiers will be '_0', '_1', etc.

    matching_methods : list or tuple, or None
        A list/tuple of matching methods that correspond to `catalog_instances`.
        Each matching method can be either a column name (usually a string),
        `MATCHING_FORMAT`, or `MATCHING_ORDER`.
        If set to `MATCHING_FORMAT`, the reader assumes the catalog instance has
        *exactly* the same underlying data format (i.e., same row order,
        same iteration over data chucks, and same native filters).
        If set to `MATCHING_ORDER`, the reader assumes the catalog instance has
        the same row order but may not have the same iteration over data chunks.
        If set to a string, the reader will use that column to match to the
        master catalog and do a left join.

    only_use_master_attr : bool, optional (default: False)
        If set to False, add-on catalogs will overwrite master catalog's
        attributes just like how they overwrite quantities.
        If set to True, only the master catalog's attributes are inherited.

    Example
    -------
    >>> cat0.list_all_quantities()
    ['a', 'b', 'c', 'd']
    >>> cat1.list_all_quantities()
    ['a', 'b', 'e']
    >>> cat2.list_all_quantities()
    ['a', 'c', 'f']

    >>> cc = CompositeCatalog([cat0, cat1, cat2])

    >>> cc.get_quantity_modifier('a')
    ('_2', 'a')
    >>> cc.get_quantity_modifier('b')
    ('_1', 'b')
    >>> cc.get_quantity_modifier('c')
    ('_2', 'c')
    >>> cc.get_quantity_modifier('d')
    ('_0', 'd')
    >>> cc.get_quantity_modifier('e')
    ('_1', 'e')
    >>> cc.get_quantity_modifier('f')
    ('_2', 'f')
    """
    def __init__(self, catalog_instances, catalog_identifiers=None,
                 matching_methods=None, only_use_master_attr=False, **kwargs):
        warnings.warn('CompositeCatalog is still an experimental feature. Use with care!')
        self._catalogs = list()
        for i, (instance, identifier, matching_method) in enumerate(
                zip_longest(
                    catalog_instances,
                    catalog_identifiers or [],
                    matching_methods or [],
                )):
            self._catalogs.append(
                CatalogWrapper(
                    instance,
                    identifier or '_{}'.format(i),
                    matching_method or MATCHING_FORMAT,
                    is_master=(i == 0),
                ))
        self._catalogs = tuple(self._catalogs)

        # check number of catalogs
        if len(self._catalogs) < 2:
            raise ValueError('Must have more than one catalogs to make a composite catalog!')

        # check uniqueness of identifiers
        identifiers = [cat.identifier for cat in self._catalogs]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError('Catalog identifiers need to be all distinct!')

        # create handy attributes for the master catalog
        self._master = self._catalogs[0]
        self.master = self._master.instance  # for users' convenience

        # check if it's possbile to do index matching
        if any(cat.need_index_matching for cat in self._catalogs[1:]):
            self._need_index_matching = True
            if not self._master.matching_column:
                raise ValueError('Must specify the column for the master catalog to do catalog matching!')
        else:
            self._need_index_matching = False

        self._need_order_matching = any(cat.matching_order for cat in self._catalogs[1:])

        self._native_filter_quantities = set(self.master.native_filter_quantities)
        self.native_filter_string_only = self.master.native_filter_string_only

        self._native_quantities = set()
        self._quantity_modifiers = dict()
        for catalog in self._catalogs:
            for q in catalog.instance.list_all_quantities(True):
                key = (catalog.identifier, q)
                self._native_quantities.add(key)
                self._quantity_modifiers[q] = key

        self.only_use_master_attr = bool(only_use_master_attr)

        super(CompositeCatalog, self).__init__(**kwargs)

    def _subclass_init(self, **kwargs):
        pass

    def _generate_native_quantity_list(self):
        return self._native_quantities

    def _obtain_native_data_dict(self, native_quantities_needed, native_quantity_getter):
        native_quantities_needed_dict = defaultdict(set)
        for identifier, quantity in native_quantities_needed:
            native_quantities_needed_dict[identifier].add(quantity)

        if self._need_index_matching:
            for catalog in self._catalogs:
                if catalog.is_master or catalog.need_index_matching:
                    native_quantities_needed_dict[catalog.identifier].add(catalog.matching_column)

        master_dummy_column = None
        if self._need_order_matching:
            if native_quantities_needed_dict[self._master.identifier]:
                master_dummy_column = list(native_quantities_needed_dict[self._master.identifier]).pop()
            else:
                master_dummy_column = list(self.master.list_all_quantities(True)).pop()
                native_quantities_needed_dict[self._master.identifier].add(master_dummy_column)

        data = dict()
        for catalog in self._catalogs:
            if catalog.identifier not in native_quantities_needed_dict:
                continue
            if catalog.matching_format:
                if native_quantity_getter.get(catalog.identifier) is None:
                    raise RuntimeError('Catalog {} does not have matching format!'.format(catalog.identifier))
                for q, v in catalog.instance._load_quantities( # pylint: disable=W0212
                        native_quantities_needed_dict[catalog.identifier],
                        native_quantity_getter[catalog.identifier],
                ).items():
                    data[(catalog.identifier, q)] = v
            elif catalog.cache is None:
                catalog.cache = catalog.instance.get_quantities(native_quantities_needed_dict[catalog.identifier])
                if catalog.matching_order:
                    catalog.counter = 0
                elif catalog.sorter is None:
                    catalog.sorter = catalog.cache[catalog.matching_column].argsort()

        for catalog in self._catalogs:
            if catalog.identifier not in native_quantities_needed_dict or catalog.matching_format:
                continue

            if catalog.matching_order:
                count = len(data[(self._master.identifier, master_dummy_column)])
                slice_this = slice(catalog.counter, catalog.counter + count)
                catalog.counter += count
                for q in native_quantities_needed_dict[catalog.identifier]:
                    data_this = catalog.cache[q][slice_this]
                    data[(catalog.identifier, q)] = data_this
                continue

            s = np.searchsorted(
                a=catalog.cache[catalog.matching_column],
                v=data[(self._master.identifier, self._master.matching_column)],
                sorter=catalog.sorter,
            )
            s[s >= len(catalog.sorter)] = -1
            matching_idx = catalog.sorter[s]
            not_matched_mask = (catalog.cache[catalog.matching_column][matching_idx] !=
                                data[(self._master.identifier, self._master.matching_column)])

            for q in native_quantities_needed_dict[catalog.identifier]:
                data_this = catalog.cache[q][matching_idx]
                if not_matched_mask.any():
                    data_this = np.ma.array(data_this, mask=not_matched_mask)
                data[(catalog.identifier, q)] = data_this

        return data

    def _iter_native_dataset(self, native_filters=None):
        for catalog in self._catalogs:
            catalog.clear()
            if catalog.matching_format:
                catalog.iterator = catalog.instance._iter_native_dataset(native_filters) # pylint: disable=W0212

        for master_data in self._master.iterator:
            dataset = {self._master.identifier: master_data}
            for catalog in self._catalogs[1:]:
                if catalog.matching_format:
                    dataset[catalog.identifier] = next(catalog.iterator, None)
            yield dataset

    def __getattr__(self, name):
        if not self.only_use_master_attr:
            for catalog in reversed(self._catalogs):
                if hasattr(catalog.instance, name):
                    return getattr(catalog.instance, name)
        return getattr(self.master, name)
