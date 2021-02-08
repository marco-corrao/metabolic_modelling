"""An API module for initializing the cache."""
# The MIT License (MIT)
#
# Copyright (c) 2013 Weizmann Institute of Science.
# Copyright (c) 2018 Institute for Molecular Systems Biology, ETH Zurich.
# Copyright (c) 2018, 2019 Novo Nordisk Foundation Center for Biosustainability,
# Technical University of Denmark
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import logging

from sqlalchemy import create_engine

from . import DEFAULT_COMPOUND_CACHE_FNAME, DEFAULT_ZENODO_DOI
from .compound_cache import CompoundCache
from .zenodo import get_cached_filepath


logger = logging.getLogger(__name__)


def create_compound_cache_from_sqlite_file(path: str) -> CompoundCache:
    """
    Initialize a compound cache from a local SQLite file.

    Parameters
    ----------
    path : str
        The path to the SQLite file.

    """
    return CompoundCache(create_engine(f"sqlite:///{path}"))


def create_compound_cache_from_zenodo(
    doi: str = DEFAULT_ZENODO_DOI,
    fname: str = DEFAULT_COMPOUND_CACHE_FNAME,
) -> CompoundCache:
    """
    Initialize a compound cache from Zenodo.

    Parameters
    ----------
    doi : str, optional
        The DOI of the Zenodo entry
        (Default value = `10.5281/zenodo.4128543`)
    fname : str, optional
        The filename of the SQLite database stored in Zenodo
        (Default value = `compounds.sqlite`).

    Returns
    -------
    CompoundCache
        a compound cache object.

    """

    path = get_cached_filepath(doi, fname)
    return create_compound_cache_from_sqlite_file(path)
