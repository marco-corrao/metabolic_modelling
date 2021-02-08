# The MIT License (MIT)
#
# Copyright (c) 2013 The Weizmann Institute of Science.
# Copyright (c) 2018 Novo Nordisk Foundation Center for Biosustainability,
# Technical University of Denmark.
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


from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

PROTON_INCHI = "InChI=1S/p+1"
WATER_INCHI = "InChI=1S/H2O/h1H2"
PROTON_INCHI_KEY = "GPRLSGONYQIRFK-UHFFFAOYSA-N"
WATER_INCHI_KEY = "XLYOFNOQVPJJNP-UHFFFAOYSA-N"

DEFAULT_COMPOUND_CACHE_FNAME = "compounds.sqlite"
DEFAULT_ZENODO_DOI = "10.5281/zenodo.4128543"

from .exceptions import ParseException
from .models import (
    Compound,
    CompoundIdentifier,
    CompoundMicrospecies,
    Base,
    Registry,
    MagnesiumDissociationConstant,
)
from .thermodynamic_constants import (
    ureg,
    Q_,
    R,
    default_T,
    FARADAY,
)
from .compound_cache import CompoundCache
from .reaction import Reaction
from .api import create_compound_cache_from_zenodo, create_compound_cache_from_sqlite_file
