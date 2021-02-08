# The MIT License (MIT)
#
# Copyright (c) 2018 Institute for Molecular Systems Biology, ETH Zurich
# Copyright (c) 2018 Novo Nordisk Foundation Center for Biosustainability,
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


from sqlalchemy.ext.declarative import declarative_base
from periodictable import elements


Base = declarative_base()
SYMBOL_TO_CHARGE = {e.symbol: e.number for e in elements}
SYMBOL_TO_CHARGE["e-"] = -1


from .mixins import TimeStampMixin
from .registry import Registry
from .compound import Compound
from .compound_identifier import CompoundIdentifier
from .compound_microspecies import CompoundMicrospecies
from .magnesium_dissociation_constant import MagnesiumDissociationConstant


__all__ = (
    "Base",
    "Registry",
    "Compound",
    "CompoundIdentifier",
    "CompoundMicrospecies",
    "MagnesiumDissociationConstant",
)
