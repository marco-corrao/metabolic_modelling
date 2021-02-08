"""A Magnesium ion dissociation constant module."""
# The MIT License (MIT)
#
# Copyright (c) 2020 Institute for Molecular Systems Biology, ETH Zurich
# Copyright (c) 2020 Novo Nordisk Foundation Center for Biosustainability,
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


from sqlalchemy import Column, Float, ForeignKey, Integer

from . import Base
from .compound import Compound
from .mixins import TimeStampMixin


class MagnesiumDissociationConstant(TimeStampMixin, Base):
    """Model a Mg2+ dissociation constant of a certain pseudoisomer."""

    __tablename__ = "magnesium_dissociation_constant"

    # SQLAlchemy column descriptors.
    id: int = Column(Integer, primary_key=True, autoincrement=True)
    compound_id: int = Column(Integer, ForeignKey(Compound.id), nullable=False)
    number_protons: int = Column(Integer, default=0, nullable=False)
    number_magnesiums: int = Column(Integer, default=0, nullable=False)
    dissociation_constant: float = Column(Float, default=None, nullable=False)

    def __repr__(self):
        """Return a string representation of this object."""
        return (
            f"{type(self).__name__}(compound_id={self.compound_id}, "
            f"number_protons={self.number_protons}, "
            f"number_magnesiums={self.number_magnesiums})"
        )
