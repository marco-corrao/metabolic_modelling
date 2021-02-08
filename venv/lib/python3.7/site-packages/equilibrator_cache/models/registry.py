"""A compound registry module."""
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
import re
import typing

from sqlalchemy import Boolean, Column, Integer, String
from sqlalchemy.orm import reconstructor, validates

from . import Base
from .mixins import TimeStampMixin


logger = logging.getLogger(__name__)


class Registry(TimeStampMixin, Base):
    r"""
    Model a MIRIAM registry resource.

    Attributes
    ----------
    id : int
        The primary key in the table.
    name : str
        The registry's common name.
    namespace : str
        The MIRIAM namespace identifier, e.g., 'metanetx.chemical'.
    pattern : str
        The regular expression pattern to validate against identifiers in the
        registry, e.g., ``^(MNXM\d+|BIOMASS)$``.
    identifier : str
        The MIRIAM registry identifier for itself, e.g., MIR:00000567.
    url : str
        The universal resource identifier (URI) for this resource, e.g.,
        'http://identifiers.org/metanetx.chemical/'.
    is_prefixed : bool
        Whether or not identifiers of this registry are prefixed with the
        ``namespace`` itself, e.g., 'CHEBI:52971'.
    access_url : str
        One of the potentially multiple URLs to access an identifier in the
        registry, e.g.,
        'http://sabiork.h-its.org/newSearch?q=sabiocompoundid:{$id}'.

    """

    __tablename__ = "registries"

    # SQLAlchemy column descriptors.
    id: int = Column(Integer, primary_key=True)
    name: typing.Optional[str] = Column(String, nullable=True)
    namespace: str = Column(String, nullable=False, index=True, unique=True)
    pattern: str = Column(String, nullable=False)
    identifier: typing.Optional[str] = Column(String, nullable=True)
    url: typing.Optional[str] = Column(String, nullable=True)
    is_prefixed: typing.Optional[bool] = Column(
        Boolean, default=False, nullable=False
    )
    access_url: typing.Optional[str] = Column(String, nullable=True)

    # Normal Python class variables.
    _identifier_pattern: typing.ClassVar[typing.Pattern] = re.compile(
        r"^MIR:\d{8}$"
    )

    def __init__(self, **kwargs):
        """Compile the identifier pattern."""
        super().__init__(**kwargs)
        self.compiled_pattern = re.compile(self.pattern)

    @reconstructor
    def init_on_load(self):
        """Compile the identifier pattern on load from database."""
        self.compiled_pattern = re.compile(self.pattern)

    def __repr__(self) -> str:
        """Return a string representation of this object."""
        return f"{type(self).__name__}(namespace={self.namespace})"

    @validates("identifier")
    def validate_identifier(self, _, identifier: str) -> str:
        """Validate the MIRIAM identifier against the pattern."""
        if self._identifier_pattern.match(identifier) is None:
            raise ValueError(
                f"The registry's identifier '{identifier}' does not match the "
                f"official pattern '^MIR:\\d{{8}}$'."
            )
        return identifier

    def is_valid_accession(self, accession: str):
        """Validate an identifier against the compiled pattern."""
        return self.compiled_pattern.match(accession) is not None
