"""
Contains the base class for a generic catalog (BaseGenericCatalog).
"""
__all__ = ['BaseGenericCatalog', 'dict_to_numpy_array', 'GCRQuery']
__version__ = '0.5.1'
__author__ = 'Yao-Yuan Mao'

import warnings
from collections import defaultdict
import numpy as np
from numpy.core.records import fromarrays
import easyquery


def _is_string_like(obj):
    """
    Check whether obj behaves like a string.
    """
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True


class GCRQuery(easyquery.Query):
    @staticmethod
    def _get_table_len(table):
        return len(next(table.values()))

    @staticmethod
    def _mask_table(table, mask):
        return {k: v[mask] for k, v in table.items()}


def _trivial_callable(x):
    return x


def dict_to_numpy_array(d):
    """
    Convert a dict of 1d array to a numpy recarray
    """
    return fromarrays(d.values(), np.dtype([(str(k), v.dtype) for k, v in d.items()]))


class BaseGenericCatalog(object):
    """
    Abstract base class for all catalog classes.
    """
    _required_attributes = set()
    _required_quantities = set()

    _default_quantity_modifier = None
    _quantity_modifiers = dict()
    _native_filter_quantities = set()

    def __init__(self, **kwargs):
        self._init_kwargs = kwargs.copy()
        self._subclass_init(**kwargs)
        self._native_quantities = set(self._generate_native_quantity_list())

        # enforce the existence of required attributes
        if not all(hasattr(self, attr) for attr in self._required_attributes):
            raise ValueError("Any subclass of BaseGenericCatalog must implement following attributes: {0}".format(', '.join(self._required_attributes)))

        # enforce the minimal set of quantities
        if not self.has_quantities(self._required_quantities):
            raise ValueError("Catalog must have the following quantities: {0}".format(self._required_quantities))

        # to check if all native quantities in the modifiers are present
        self._check_quantities_exist(self.list_all_quantities(True), raise_exception=False)


    def get_quantities(self, quantities, filters=None, native_filters=None, return_iterator=False):
        """
        Fetch quantities from this galaxy catalog.

        Parameters
        ----------
        quantities : str or list of str or tuple of str
            quantities to fetch

        filters : list of tuple, or GCRQuery instance, optional
            filters to apply. Each filter should be in the format of (callable, str, str, ...)

        native_filters : list of tuple, optional
            Native filters to apply. Each filter should be in the format of (callable, str, str, ...)

        return_iterator : bool, optional
            if True, return an iterator that iterates over the native format, default is False

        Returns
        -------
        quantities : dict, or iterator of dict (when `return_iterator` is True)
        """

        quantities = self._preprocess_requested_quantities(quantities)
        filters = self._preprocess_filters(filters)
        if native_filters:
            self._check_native_filters(native_filters)

        it = self._get_quantities_iter(quantities, filters, native_filters)

        if return_iterator:
            return it

        data_all = defaultdict(list)
        for data in it:
            for q in quantities:
                data_all[q].append(data[q])
        return {q: (np.concatenate(data_all[q]) if len(data_all[q]) > 1 else data_all[q][0]) for q in quantities}


    def has_quantity(self, quantity, include_native=True):
        """
        Check if *quantity* is available in this galaxy catalog

        Parameters
        ----------
        quantity : str
            a quantity name to check

        include_native : bool, optional
            whether or not to include native quantity names when checking

        Returns
        -------
        has_quantity : bool
            True if the quantities are all available; otherwise False
        """

        if include_native:
            return all(q in self._native_quantities for q in self._translate_quantities({quantity}))

        return quantity in self._quantity_modifiers


    def has_quantities(self, quantities, include_native=True):
        """
        Check if ALL *quantities* specified are available in this galaxy catalog

        Parameters
        ----------
        quantities : iterable
            a list of quantity names to check

        include_native : bool, optional
            whether or not to include native quantity names when checking

        Returns
        -------
        has_quantities : bool
            True if the quantities are all available; otherwise False
        """
        quantities = set(quantities)

        if include_native:
            return all(q in self._native_quantities for q in self._translate_quantities(quantities))

        return all(q in self._quantity_modifiers for q in quantities)


    def list_all_quantities(self, include_native=False):
        """
        Return a list of all available quantities in this catalog
        """
        output = list(self._quantity_modifiers)
        if include_native:
            for q in self._native_quantities:
                if q not in output:
                    output.append(q)
        return output


    def list_all_native_quantities(self):
        """
        Return a list of all available native quantities in this catalog
        """
        return list(self._native_quantities)


    def first_available(self, *quantities):
        """
        Return the first available quantity in the input arguments.
        Return `None` if none of them is available.
        """
        for i, q in enumerate(quantities):
            if self.has_quantity(q):
                if i:
                    warnings.warn('{} not available; using {} instead'.format(quantities[0], q))
                return q


    def get_input_kwargs(self, key=None):
        """
        Get the input keyword arguments.
        If *key* is `None`, return the full dict.
        """
        if key is None:
            return self._init_kwargs

        return self._init_kwargs.get(key)


    def add_quantity_modifier(self, quantity, modifier, overwrite=False):
        """
        Add a quantify modifier.

        Parameters
        ----------
        quantity : str
            name of the derived quantity to add

        modifier : None or str or tuple
            If the quantity modifier is a tuple of length >=2 and the first element is a callable,
            it should be in the formate of `(callable, native quantity 1,  native quantity 2, ...)`.
            And the modifier would work as callable(native quantity 1,  native quantity 2, ...)
            If the quantity modifier is None, the quantity will be used as the native quantity name
            Otherwise, the modifier would be use directly as a native quantity name

        overwrite : bool, optional
            If False and quantity are already specified in _quantity_modifiers, raise an ValueError
        """
        if quantity in self._quantity_modifiers and not overwrite:
            raise ValueError('quantity `{}` already exists'.format(quantity))
        self._quantity_modifiers[quantity] = modifier
        self._check_quantities_exist([quantity], raise_exception=False)


    def get_quantity_modifier(self, quantity):
        """
        Retrive a quantify modifier.

        Parameters
        ----------
        quantity : str
            name of the derived quantity to get

        Returns
        -------
        quantity_modifier
        """
        return self._quantity_modifiers.get(quantity, self._default_quantity_modifier)


    def get_normalized_quantity_modifier(self, quantity):
        """
        Retrive a quantify modifier, normalized.
        This function would also return a tuple, with the first item a callable,
        and the rest native quantity names

        Parameters
        ----------
        quantity : str
            name of the derived quantity to get

        Returns
        -------
        tuple : (callable, quantity1, quantity2...)
        """
        modifier = self._quantity_modifiers.get(quantity, self._default_quantity_modifier)
        if modifier is None:
            return (_trivial_callable, quantity)

        if callable(modifier):
            return (modifier, quantity)

        if isinstance(modifier, (tuple, list)) and len(modifier) > 1 and callable(modifier[0]):
            return modifier

        return (_trivial_callable, modifier)


    def add_modifier_on_derived_quantities(self, new_quantity, func, *quantities):
        """
        Add a quantify modifier.

        Parameters
        ----------
        new_quantity : str
            name of the new quantity to add

        func : callable

        quantities : list of str
            quantities to pass to the callable
        """
        if new_quantity in self._quantity_modifiers:
            raise ValueError('quantity name `{}` already exists'.format(new_quantity))

        functions = []
        quantities_needed = []
        quantity_count = []
        for q in quantities:
            modifier = self.get_normalized_quantity_modifier(q)
            functions.append(modifier[0])
            quantities_needed.extend(modifier[1:])
            quantity_count.append(len(modifier)-1)

        def _new_func(*x):
            assert len(x) == sum(quantity_count)
            count_current = 0
            new_args = []
            for func_this, count in zip(functions, quantity_count):
                new_args.append(func_this(*x[count_current:count_current+count]))
                count_current += count
            return func(*new_args)

        self._quantity_modifiers[new_quantity] = tuple([_new_func] + quantities_needed)
        self._check_quantities_exist([new_quantity], raise_exception=False)


    def del_quantity_modifier(self, quantity):
        """
        Delete a quantify modifier.

        Parameters
        ----------
        quantity : str
            name of the derived quantity to delete
        """
        if quantity in self._quantity_modifiers:
            del self._quantity_modifiers[quantity]


    def _translate_quantity(self, quantity_requested, native_quantities_needed=None):
        if native_quantities_needed is None:
            native_quantities_needed = defaultdict(list)

        modifier = self._quantity_modifiers.get(quantity_requested, self._default_quantity_modifier)

        if modifier is None or callable(modifier):
            return native_quantities_needed[quantity_requested].append(quantity_requested)

        elif isinstance(modifier, (tuple, list)) and len(modifier) > 1 and callable(modifier[0]):
            for native_quantity in modifier[1:]:
                native_quantities_needed[native_quantity].append(quantity_requested)

        else:
            native_quantities_needed[modifier].append(quantity_requested)

        return native_quantities_needed


    def _translate_quantities(self, quantities_requested):
        native_quantities_needed = defaultdict(list)

        for q in quantities_requested:
            self._translate_quantity(q, native_quantities_needed)

        return native_quantities_needed


    def _check_quantities_exist(self, quantities_requested, raise_exception=False):
        for native_quantity, quantities in self._translate_quantities(quantities_requested).items():
            if native_quantity not in self._native_quantities:
                msg = 'Native quantity `{}` does not exist (required by `{}`)'.format(native_quantity, '`, `'.join(quantities))
                if raise_exception:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg)
                    return False
        return True


    def _preprocess_requested_quantities(self, quantities):
        if _is_string_like(quantities):
            quantities = {quantities}

        quantities = set(quantities)
        if not quantities:
            raise ValueError('You must set `quantities`.')

        self._check_quantities_exist(quantities, raise_exception=True)

        return quantities


    def _preprocess_filters(self, filters):
        if filters is None:
            filters = GCRQuery()
        elif _is_string_like(filters):
            filters = GCRQuery(filters)
        else:
            filters = GCRQuery(*filters)

        self._check_quantities_exist(filters.variable_names, raise_exception=True)

        return filters


    def _check_native_filters(self, native_filters):
        for f in native_filters:
            if isinstance(f, (tuple, list)) and \
                len(f) > 1 and \
                callable(f[0]) and \
                set(f[1:]).issubset(self._native_filter_quantities):
                continue

            raise ValueError('`native_filters` is not set correctly. Must be None or [(callable, str, str, ...), ...]')


    def _assemble_quantity(self, quantity_requested, native_quantities_loaded):
        modifier = self._quantity_modifiers.get(quantity_requested, self._default_quantity_modifier)

        if modifier is None:
            return native_quantities_loaded[quantity_requested]

        elif callable(modifier):
            return modifier(native_quantities_loaded[quantity_requested])

        elif isinstance(modifier, (tuple, list)) and len(modifier) > 1 and callable(modifier[0]):
            return modifier[0](*(native_quantities_loaded[_] for _ in modifier[1:]))

        return native_quantities_loaded[modifier]


    def _load_quantities(self, quantities, native_quantity_getter):
        native_data = {q: native_quantity_getter(q) for q in self._translate_quantities(quantities)}
        return {q: self._assemble_quantity(q, native_data) for q in quantities}


    def _get_quantities_iter(self, quantities, filters, native_filters):
        for native_quantity_getter in self._iter_native_dataset(native_filters):
            data = self._load_quantities(quantities.union(set(filters.variable_names)),
                                         native_quantity_getter)
            data = filters.filter(data)
            for q in set(data).difference(quantities):
                del data[q]
            yield data
            del data


    def __getitem__(self, key):
        return self.get_quantities([key])[key]


    def _subclass_init(self, **kwargs):
        """
        To be implemented by subclass.
        Must return `None`.
        Must accept any keyword argument (i.e., must have **kwargs).
        This method is called during __init__().
        """
        raise NotImplementedError


    def _generate_native_quantity_list(self):
        """
        To be implemented by subclass.
        Must return an iterable of all native quantity names.
        """
        raise NotImplementedError


    def _iter_native_dataset(self, native_filters=None):
        """
        To be implemented by subclass.
        Must be a generator.
        Must yield a callable, *native_quantity_getter*.
        This function must iterate over subsets of rows, not columns!

        Below are specifications of *native_quantity_getter*
        -----------------------------------------
        Must take a single argument of a native quantity name.
        Should assume the argument is valid.
        Must return a numpy 1d array.
        """
        raise NotImplementedError
