"""A compound identifier module."""
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


import logging

from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from . import Base
from .compound import Compound
from .mixins import TimeStampMixin
from .registry import Registry


logger = logging.getLogger(__name__)


class CompoundIdentifier(TimeStampMixin, Base):
    """Model a compound's identifiers in various namespaces.

    Attributes
    ----------
    id : int
        The primary key in the table.
    compound_id : int
        The foreign key of the related compound.
    registry_id : int
        The foreign key of the related registry.
    registry : Registry
        The identifier's namespace registry in a one-to-one relationship.
    accession : str
        The identifying string.

    """

    __tablename__ = "compound_identifiers"

    # SQLAlchemy column descriptors.
    id: int = Column(Integer, primary_key=True, autoincrement=True)
    compound_id: int = Column(Integer, ForeignKey(Compound.id), nullable=False)
    registry_id: int = Column(Integer, ForeignKey(Registry.id), nullable=False)
    registry: Registry = relationship(Registry, lazy="selectin")
    accession: str = Column(String, nullable=False, index=True)

    def __repr__(self) -> str:
        """Return a string representation of this object."""
        return (
            f"{type(self).__name__}(registry={repr(self.registry)},"
            f" accession={self.accession})"
        )

    def is_valid(self) -> bool:
        """Use the registry to validate the accession."""
        if self.registry is None:
            logger.error("No associated registry.")
            return False
        if not self.registry.is_valid_accession(self.accession):
            logger.error(
                f"Identifier '{self.accession}' does not match "
                f"{self.registry.name}'s pattern '{self.registry.pattern}'."
            )
            return False
        return True
