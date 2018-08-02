"""
GCRQuery module (a subclass of easyquery.Query)
"""
import numpy as np
import easyquery
from .utils import is_string_like

__all__ = ['GCRQuery']

class GCRQuery(easyquery.Query):
    @staticmethod
    def _get_table_len(table):
        try:
            return len(next(iter(table.values())))
        except StopIteration:
            return 0

    @staticmethod
    def _mask_table(table, mask):
        return {k: v[mask] for k, v in table.items()}

    @staticmethod
    def _check_basic_query(basic_query):
        return basic_query is None or is_string_like(basic_query) or \
                (isinstance(basic_query, tuple) and \
                len(basic_query) > 1 and callable(basic_query[0]))

    def check_scalar(self, scalar_dict):
        """
        check if `scalar_dict` satisfy query
        """
        table = {k: np.array([v]) for k, v in scalar_dict.items()}
        return self.mask(table)[0]
