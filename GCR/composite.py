"""
Composite catalog reader
"""
import warnings
from collections import defaultdict
from collections.abc import Mapping
from itertools import repeat

import numpy as np

from .base import BaseGenericCatalog

__all__ = ['CompositeSpecs', 'CompositeCatalog', 'MATCHING_FORMAT', 'MATCHING_ORDER']


# For backward compatibility: define module constants to be used in matching_columns
MATCHING_FORMAT = None
MATCHING_ORDER = tuple()


def _match(index_this, index_main, sorter=None):
    if sorter is None:
        sorter = np.argsort(index_this)
    s = np.searchsorted(index_this, index_main, sorter=sorter)
    s[s >= len(sorter)] = -1
    matching_idx = sorter[s]
    not_matched_mask = (index_this[matching_idx] != index_main)
    return matching_idx, not_matched_mask, sorter


def _slice_matched(input_data_dict, quantity_needed, matching_idx, not_matched_mask,
                   always_return_masked_array=False):
    for q in quantity_needed:
        data_this = input_data_dict[q][matching_idx]
        if always_return_masked_array or not_matched_mask.any():
            data_this = np.ma.array(data_this, mask=not_matched_mask)
        yield q, data_this


class CompositeSpecs(object):
    """
    CompositeSpecs class

    Parameters
    ----------
    instance : instance of GCR.BaseGenericCatalog
    identifier : str, name of this catalog
    matching_partition : bool, optional (default: True)
        Whether this catalog has matching partitions as the main catalog
    matching_row_order : bool, optional (default: True)
        Whether this catalog has the same row ordering as the main catalog
    matching_by_column : str, optional (default: None)
        The column in this catalog to be used to match to the main catalog.
        If set, matching_row_order is assumed to be False.
    matching_column_in_main : str, optional (default: None)
        The column in the main catalog to be used to match to this catalog.
    overwrite_quantities : bool (default: True)
        Whether this catalog should overwrite existing columns in the main catalog
    overwrite_attributes : bool (default: True)
        Whether this catalog should overwrite existing catalog attributes in the main catalog
    include_native_quantities : bool (default: True)
        Whether native quantities should be included
    is_main : bool (default: False)
        Whether this catalog is the main catalog
    """
    def __init__(
        self,
        instance,
        identifier=None,
        matching_partition=True,
        matching_row_order=True,
        matching_by_column=None,
        matching_column_in_main=None,
        overwrite_quantities=True,
        overwrite_attributes=True,
        include_native_quantities=True,
        is_main=False,
        **kwargs
    ):
        self.instance = instance
        self.identifier = str(identifier or "")

        self.matching_partition = bool(matching_partition)
        self.matching_row_order = bool(matching_row_order)
        self.matching_by_column = self.matching_column_in_main = None
        if matching_by_column:
            self.set_matching_column(matching_by_column, matching_column_in_main, matching_partition)

        # For backward compatibility
        if "matching_method" in kwargs:
            matching_method = kwargs["matching_method"]
            if matching_method == MATCHING_FORMAT or matching_method == "MATCHING_FORMAT":
                self.set_matching_format()
            elif matching_method == MATCHING_ORDER or matching_method == "MATCHING_ORDER":
                self.set_matching_order()
            else:
                self.set_matching_column(matching_method)

        self.overwrite_quantities = bool(overwrite_quantities)
        self.overwrite_attributes = bool(overwrite_attributes)
        self.include_native_quantities = bool(include_native_quantities)

        self._is_main = None
        self.is_main = is_main

        self.other_kwargs = kwargs.copy()

        self.cache = None
        self.sorter = None
        self.counter = None

    def clear(self):
        """
        clear cache
        """
        self.cache = None
        self.counter = None

    @property
    def is_main(self):
        return self._is_main

    @is_main.setter
    def is_main(self, value):
        self._is_main = bool(value)
        if self._is_main:
            self.set_matching_format()
            self.overwrite_quantities = True
            self.overwrite_attributes = True

    def set_matching_format(self):
        self.matching_partition = True
        self.matching_row_order = True
        self.matching_by_column = None
        self.matching_column_in_main = None

    def set_matching_order(self):
        self.matching_partition = False
        self.matching_row_order = True
        self.matching_by_column = None
        self.matching_column_in_main = None

    def set_matching_column(self, column, column_in_main=None, same_partition=False):
        self.matching_partition = bool(same_partition)
        self.matching_row_order = False
        self.matching_by_column = column
        self.matching_column_in_main = column_in_main if column_in_main else column

    @property
    def is_valid_matching(self):
        return bool(self.matching_row_order or self.matching_by_column)

    def get_data_iterator(self, native_filters=None):
        self.clear()
        if self.matching_partition:
            return self.instance._iter_native_dataset(native_filters)
        return repeat(None)


class CompositeCatalog(BaseGenericCatalog):
    """
    Composite Catalog class

    Parameters
    ----------
    catalog_instances : A list of CompositeSpecs instance.
        Alternatively, a list of dictionaries that can be used to init CompositeSpecs.
        See CompositeSpecs docstring for details.
    always_return_masked_array : bool, optional (default: False)
        If set to True, always return masked array unless the catalogs have matching formats.

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
    def __init__(
        self,
        catalog_instances,
        catalog_identifiers=None,
        matching_methods=None,
        only_use_master_attr=None,
        always_return_masked_array=False,
        **kwargs
    ):
        warnings.warn('CompositeCatalog is still an experimental feature. Use with care!')

        # check number of catalogs
        if len(catalog_instances) < 2:
            raise ValueError('Must have more than one catalogs to make a composite catalog!')

        self._catalogs = list()
        for i, instance in enumerate(catalog_instances):
            if isinstance(instance, CompositeSpecs):
                cat = instance

            elif isinstance(instance, Mapping):
                cat = CompositeSpecs(**instance)

            else:
                warnings.warn("Please supply a list of CompositeSpecs instances", DeprecationWarning)
                try:
                    identifier = catalog_identifiers[i]
                except (TypeError, KeyError, IndexError):
                    identifier = None
                try:
                    matching_method = matching_methods[i]
                except (TypeError, KeyError, IndexError):
                    matching_method = MATCHING_FORMAT
                cat = CompositeSpecs(instance, identifier, matching_method=matching_method)

            self._catalogs.append(cat)

        # check uniqueness of main catalogs
        main_flags = [cat.is_main for cat in self._catalogs]
        if sum(main_flags) > 1:
            raise ValueError('There can be only one main catalog!')
        elif not any(main_flags):
            self._main = self._catalogs[0]
            self._main.is_main = True
        else:
            # make sure main catalog is the first catalog
            self._main = self._catalogs.pop(main_flags.index(True))
            self._catalogs.insert(0, self._main)

        self._catalogs = tuple(self._catalogs)

        # check uniqueness of identifiers
        identifiers = [cat.identifier for cat in self._catalogs]
        for i, cat in enumerate(self._catalogs):
            if identifiers.index(cat.identifier) != i or not cat.identifier:
                cat.identifier = cat.identifier + "_{}".format(i)

        # check all catalogs have valid matching method
        if not all(cat.is_valid_matching for cat in self._catalogs):
            raise ValueError("Not all catalogs have valid matching method!")

        # backward compatibility: matching_column_in_main was used to set in main catalog's matching_method
        main_matching_col = self._main.other_kwargs.get("matching_method")
        if main_matching_col and self._main.instance.has_quantity(main_matching_col):
            for cat in self._catalogs:
                if (
                    cat.matching_column_in_main and
                    cat.matching_column_in_main == cat.other_kwargs.get("matching_method")
                ):
                    cat.matching_column_in_main = main_matching_col

        self._native_filter_quantities = set(self.main.native_filter_quantities)
        self.native_filter_string_only = self.main.native_filter_string_only

        self._native_quantities = set()
        self._quantity_modifiers = dict()
        for cat in self._catalogs:
            for q in cat.instance.list_all_quantities(cat.include_native_quantities):
                key = (cat.identifier, q)
                self._native_quantities.add(key)
                if cat.overwrite_quantities or q not in self._quantity_modifiers:
                    self._quantity_modifiers[q] = key

        if only_use_master_attr:
            for cat in self._catalogs[1:]:
                cat.overwrite_attributes = False

        self.always_return_masked_array = bool(always_return_masked_array)

        super(CompositeCatalog, self).__init__(**kwargs)

    def _subclass_init(self, **kwargs):
        pass

    def _generate_native_quantity_list(self):
        return self._native_quantities

    def _obtain_native_data_dict(self, native_quantities_needed, native_quantity_getter):
        native_quantities_needed_dict = defaultdict(set)
        for identifier, quantity in native_quantities_needed:
            native_quantities_needed_dict[identifier].add(quantity)

        order_matching_dummy_col = None

        for cat_id in list(native_quantities_needed_dict):
            cat = self._get_catalog_by_id(cat_id)

            if cat.is_main:
                continue

            # add matching columns
            if cat.matching_by_column:
                native_quantities_needed_dict[cat_id].add(cat.matching_by_column)
                native_quantities_needed_dict[self._main.identifier].add(cat.matching_column_in_main)

            # set order_matching_dummy_col if needed
            elif order_matching_dummy_col is None and cat.matching_row_order and not cat.matching_partition:
                if native_quantities_needed_dict[self._main.identifier]:
                    order_matching_dummy_col = list(native_quantities_needed_dict[self._main.identifier]).pop()
                else:
                    order_matching_dummy_col = list(self.master.list_all_quantities(True)).pop()
                    native_quantities_needed_dict[self._main.identifier].add(order_matching_dummy_col)

        data = dict()
        for cat in self._catalogs:
            if cat.identifier not in native_quantities_needed_dict:
                continue

            quantities_needed = native_quantities_needed_dict[cat.identifier]
            getter = native_quantity_getter[cat.identifier]

            # load data
            if cat.is_main or cat.matching_partition:
                cat.cache = cat.instance._load_quantities(quantities_needed, getter)
            elif cat.cache is None:
                cat.cache = cat.instance.get_quantities(quantities_needed)

            # for main catalog or catalog has exact matching format, import data and do nothing else:
            if cat.is_main or (cat.matching_partition and cat.matching_row_order):
                for q, v in cat.cache.items():
                    data[(cat.identifier, q)] = v
                cat.cache = None
                continue

            # match column if needed:
            if cat.matching_by_column:
                matching_idx, not_matched_mask, cat.sorter = _match(
                    cat.cache[cat.matching_by_column],
                    data[(self._main.identifier, cat.matching_column_in_main)],
                    cat.sorter,
                )

                for q, v in _slice_matched(
                    cat.cache,
                    quantities_needed,
                    matching_idx,
                    not_matched_mask,
                    self.always_return_masked_array
                ):
                    data[(cat.identifier, q)] = v

            # slice rows if needed
            if order_matching_dummy_col is not None:
                if not cat.counter:
                    cat.counter = 0
                count = len(data[(self._main.identifier, order_matching_dummy_col)])
                slice_this = slice(cat.counter, cat.counter + count)
                cat.counter += count
                for q in quantities_needed:
                    data[(cat.identifier, q)] = cat.cache[q][slice_this]

            if cat.matching_partition:
                cat.cache = cat.sorter = cat.counter = None

        return data

    def _iter_native_dataset(self, native_filters=None):
        identifiers = tuple((cat.identifier for cat in self._catalogs))
        for getters in zip(*(cat.get_data_iterator(native_filters) for cat in self._catalogs)):
            yield dict(zip(identifiers, getters))

    def __getattr__(self, name):
        attr_found = False
        attr = None
        for cat in reversed(self._catalogs):
            if hasattr(cat.instance, name):
                if cat.overwrite_attributes:
                    return getattr(cat.instance, name)
                else:
                    attr = getattr(cat.instance, name)
                    attr_found = True
        else:
            if attr_found:
                return attr
        return getattr(self.main, name)

    def _get_quantity_info_dict(self, quantity, default=None):
        native_q = self._quantity_modifiers.get(quantity)
        if isinstance(native_q, tuple) and len(native_q) == 2:
            cat_id, q = native_q
            try:
                cat = self._get_catalog_by_id(cat_id)
            except KeyError:
                pass
            else:
                return cat.instance.get_quantity_info(q, default=default)
        return default

    def _get_catalog_by_id(self, identifier):
        for cat in self._catalogs:
            if cat.identifier == identifier:
                return cat
        raise KeyError(identifier, "not exist!")

    @property
    def catalogs(self):
        return tuple((cat.instance for cat in self._catalogs))

    @property
    def main(self):
        return self.catalogs[0]

    @property
    def master(self):
        # backward compatibility
        return self.main

    def __len__(self):
        return len(self.main)
