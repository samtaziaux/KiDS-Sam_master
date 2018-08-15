"""Utilities to deal with the selection function

The selection function may come in two formats: a table or a function.
Right now only the table format is working. In this case, the 


"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.io import ascii
import numpy as np
from scipy.interpolate import griddata
import six


class Selection():
    """Sample selection object

    Read a table containing a *regular grid* of (redshift,observable)
    pairs in addition to the completeness at each coordinate. Support
    for irregular grids may be added in the future if requested.

    If the (redshift,observable) grid is not provided, the functions
    `expand_obs_range` and `expand_z_range` (used in KiDS-GGL) will
    fail. If the Selection object is defined in another context that
    does not require those methods then the condition of a regular grid
    is no longer necessary.
    """

    def __init__(self, filename, colnames=None, format=None):
        self.filename = filename
        self.format = format
        self.colnames = colnames
        self._table = None

    @property
    def colnames(self):
        return self._colnames

    @colnames.setter
    def colnames(self, colnames):
        if colnames is None:
            self._colnames = ['col0', 'col1', 'col2']
        elif isinstance(colnames, six.string_types):
            self._colnames = colnames.split(',')
        else:
            self._colnames = colnames
        assert len(self._colnames) == 3, \
            'the second entry must be a list of three column' \
            ' names, for those columns containing the redshift,' \
            ' observable, and completeness level, in that order'

    @property
    def table(self):
        if self._table is None:
            self._table = ascii.read(
                self.filename, format=self.format,
                include_names=self.colnames)
        assert np.all([c in self._table.colnames for c in self.colnames]), \
            'Not all column names present in {0}'.format(self.filename)
        return self._table

    def expand_obs_range(self, obsleft=None, obsright=None, left=0, right=1,
                         in_place=True):
        """Append values with lower and/or higher observable values

        Parameters
        ----------
        obsleft, obsright: float, optional
            observable value(s) to append at the left or right of the
            table. If neither is specified, or if neither are outside
            the range already included in the table, nothing will
            happen (including no error)
        left, right: float or None, optional
            completeness values to append at the left or right. If set
            to None, then the first or last columns will be reproduced
            at the left or right, respectively
        in_place: bool, optional
            whether to update the table in-place or return a new table
        """
        obs = self.table[self.colnames[1]]
        # do nothing
        if obsleft is None and obsright is None \
                or (obsleft >= obs.min() and obsright <= obs.max()):
            if in_place:
                return
            else:
                return self.table
        z = self.table[self.colnames[0]]
        c = self.table[self.colnames[2]]
        if obsleft is not None and obsleft < obs.min():
            if left is None:
                left = c[obs == obs.min()]
            first = Table(
                [z, [obsleft]*z.size, [left]*z.size], names=self.colnames)
            tbl = join(first, self.table)
        if obsright is not None and obsright > obs.max():
            last = Table(
                [z, [obsright]*z.size, [right]*z.size], names=self.colnames)
            tbl = join(tbl, last)
        self._table = tbl

    def expand_z_range(self, zmin=None, zmax=None, left=0, right=1,
                       in_place=True):
        """Append values with lower and/or higher redshifts

        Parameters
        ----------
        zleft, zright: float, optional
            redshift(s) to append at the left or right of the
            table. If neither is specified, or if neither are outside
            the range already included in the table, nothing will
            happen (including no error)
        left, right: float or None, optional
            completeness values to append at the left or right. If set
            to None, then the first or last columns will be reproduced
            at the left or right, respectively
        in_place: bool, optional
            whether to update the table in-place or return a new table
        """
        z = self.table[self.colnames[0]]
        # do nothing
        if zleft is None and zright is None \
                or (zleft >= z.min() and zright <= z.max()):
            if in_place:
                return
            else:
                return self.table
        obs = self.table[self.colnames[1]]
        c = self.table[self.colnames[2]]
        if zleft is not None and zleft < z.min():
            if left is None:
                left = c[z == z.min()]
            first = Table(
                [z, [zleft]*z.size, [left]*z.size], names=self.colnames)
            tbl = join(first, self.table)
        if zright is not None and zright > z.max():
            last = Table(
                [z, [zright]*z.size, [right]*z.size], names=self.colnames)
            tbl = join(tbl, last)
        self._table = tbl

    def interpolate(self, z, x, method='cubic'):
        """Interpolate the selection function

        Interpolate the selection function to the desired redshift and
        observable value(s) using `scipy.interpolate.griddata`.

        Parameters
        ----------
        z : array-like of floats, shape (N,)
            redshift values at which to interpolate
        x : array-like of floats, shape (N,)
            observable values at which to interpolate
        methohd : {'linear', 'nearest', 'cubic'}, optional
            see `scipy.interpolate.griddata` for details

        Returns
        -------
        grid : array of floats, shape (N,)
            completeness values at the (redshift,observable) locations
        """
        if self.table == 'None':
            return np.ones_like(z)
        xy = np.transpose(
            [self.table[self.colnames[0]], self.table[self.colnames[1]]])
        return griddata(xy, self.table[self.colnames[2]], np.transpose([z,x]))


def write(data, filename, format='ascii.fixed_width', **kwargs):
    """Write data to a file and return Selection object

    Parameters
    ----------
    data : `astropy.table.Table` or array-like
        if array-like, must be [z, observable, completeness] for the
        resulting file to be interpreted correctly by the Selection
        object, unless the kwarg `names` specifies all column names
    filename : str
        output file name
    format : str, optional
        `astropy` write format
    kwargs : dict, optional
        dictionary of keyword arguments passed to `Table.write`

    Returns
    -------
    sel : `Selection` object
        completeness as a function of redshift and observable
    """
    Table.write(data, filename, format=format, **kwargs)
    sel = Selection(filename, colnames, format)
    return sel


