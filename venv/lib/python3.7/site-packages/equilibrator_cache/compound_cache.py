"""This module is the interface to the compound cache database."""
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
from functools import partial
from typing import Iterable, List, Optional, Set, Tuple

import pandas as pd
import sqlalchemy
from Levenshtein import ratio
from sqlalchemy import exists
from sqlalchemy.orm import joinedload, sessionmaker
from sqlalchemy.orm.session import make_transient

from . import PROTON_INCHI_KEY, WATER_INCHI_KEY
from .models import Compound, CompoundIdentifier, CompoundMicrospecies, Registry


__all__ = ("CompoundCache",)


logger = logging.getLogger(__name__)
Session = sessionmaker()


class CompoundCache:
    """
    Implement a compound cache for look ups.

    CompoundCache is a singleton that handles caching of Compound objects for
    the component-contribution package.  The Compounds are retrieved by their
    ID (e.g., KEGG COMPOUND ID, ChEBI Id or HMDB in most cases) or InChI Key.
    The first time a Compound is requested, it is obtained from the relevant
    database and a Compound object is created. Any further
    request for the same Compound ID will draw the object from the cache. When
    the method dump() is called, all cached data is written to a file that will
    be loaded in future python sessions.

    """

    def __init__(self, engine: sqlalchemy.engine.base.Engine):
        """Initialize an in memory cache for compounds."""
        self.compound_dict = {}
        self.engine = engine
        self.session = Session(bind=self.engine)
        self._protons = None
        self._waters = None
        logger.debug("Loading synonyms into data frame.")
        query = (
            self.session.query(
                CompoundIdentifier.compound_id, CompoundIdentifier.accession
            )
            .join(Registry)
            .filter(Registry.namespace == "synonyms")
        )
        self._synonyms = pd.read_sql(
            query.statement, query.session.bind, index_col="compound_id"
        )
        self._synonyms["accession"] = self._synonyms["accession"].str.lower()

    @property
    def proton(self) -> Compound:
        """Return the H+ compound object."""

        if self._protons is None:
            self._protons = self.search_compound_by_inchi_key(PROTON_INCHI_KEY)
        return self._protons[0]

    def is_proton(self, cpd: Compound) -> bool:
        """Return True if this compound is H+."""

        self.proton  # this is used to initialize the _proton member
        return cpd in self._protons

    @property
    def water(self) -> Compound:
        """Return the H2O compound object."""

        if self._waters is None:
            self._waters = self.search_compound_by_inchi_key(WATER_INCHI_KEY)
        return self._waters[0]

    def is_water(self, cpd: Compound) -> bool:
        """Return True if this compound is H2O."""

        self.water  # this is used to initialize the _water member
        return cpd in self._waters

    def all_compound_accessions(self, ascending: bool = True) -> List[str]:
        """Return a list of all compound accessions."""

        return sorted(
            (
                row.accession
                for row in self.session.query(
                    CompoundIdentifier.accession
                ).distinct()
            ),
            reverse=not ascending,
        )

    def accession_exists(self, accession: str) -> bool:
        """Return True if this accession exists in the cache."""

        return self.session.query(
            exists().where(CompoundIdentifier.accession == accession)
        ).scalar()

    def get_compound_by_internal_id(self, compound_id: int) -> Compound:
        """Find a compound in the cache based on the internal ID."""

        return (
            self.session.query(Compound).filter_by(id=compound_id).one_or_none()
        )

    def get_compound_by_inchi(self, inchi: str) -> Optional[Compound]:
        """Return a compound by exact match of the InChI."""

        hits = (
            self.session.query(Compound)
            .filter_by(inchi=inchi)
            .options(joinedload("microspecies"))
            .all()
        )
        if len(hits) == 0:
            return None
        if len(hits) == 1:
            return hits[0]
        else:
            compound = hits[0]
            # We don't want to persist the following changes to identifiers
            # to the database. Thus we detach this object from the session.
            make_transient(compound)
            for cmpd in hits[1:]:
                compound.identifiers.extend(cmpd.identifiers)
            return compound

    def search_compound_by_inchi_key(self, inchi_key: str) -> List[Compound]:
        """Return all compounds matching the (partial) InChI Key.

        Certain parts at the end of an InChI Key may be omitted in order to
        omit specific InChI layers of information and retrieve all generally
        matching compounds.
        """
        query = self.session.query(Compound)
        if len(inchi_key) < 27:
            return query.filter(Compound.inchi_key.like(f"{inchi_key}%")).all()
        else:
            return query.filter_by(inchi_key=inchi_key).all()

    def get_compound(self, compound_id: str) -> Compound:
        """Find a compound object based on the compound ID."""

        try:
            namespace, accession = compound_id.split(":", 1)
        except ValueError:
            namespace, accession = None, compound_id
        else:
            # Special case for ChEBI identifiers.
            if namespace == "CHEBI":
                namespace = "chebi"
                accession = compound_id
            else:
                namespace = namespace.lower()

        return self.get_compound_from_registry(namespace, accession)

    def get_compound_from_registry(
        self, namespace: str, accession: str
    ) -> Compound:
        """Find a compound object by its accession (and namespace)."""

        if (namespace, accession) in self.compound_dict:
            logging.debug(f"Cache hit for {accession} in {namespace} in RAM")
            return self.compound_dict[(namespace, accession)]

        query = self.session.query(Compound)
        query = query.outerjoin(CompoundIdentifier).filter(
            CompoundIdentifier.accession == accession
        )
        if namespace is None:
            # if the namespace is not given, use the accession alone to
            # find the compound
            logging.debug(f"Looking for {accession} in all namespaces")
        else:
            # otherwise, use the specific namespace and accession to
            # locate the compound.
            logging.debug(f"Looking for {accession} in {namespace}")
            query = query.outerjoin(Registry).filter(
                Registry.namespace == namespace
            )

        compound = query.one_or_none()

        if compound is None:
            logging.debug("Cache miss")
            return None
        else:
            logging.debug("Cache hit")
            self.compound_dict[(namespace, accession)] = compound
            return compound

    @staticmethod
    def get_element_data_frame(compounds: Iterable[Compound]) -> pd.DataFrame:
        """Tabulate the elemental composition of the given compounds.

        Parameters
        ----------
        compounds : Iterable[Compound]
            A collection of compounds.

        Returns
        -------
        DataFrame
            A data frame where the columns are the compounds and the
            indexes are atomic elements.

        """

        atom_bags = {
            compound: compound.atom_bag or {} for compound in compounds
        }

        # create the elemental matrix, where each row is a compound and each
        # column is an element (or e-)
        return pd.DataFrame.from_dict(
            atom_bags, orient="columns", dtype=int
        ).fillna(0)

    def get_compound_names(self, compound: Compound) -> Set[str]:
        """Return all names for this compound."""
        query = (
            self.session.query(CompoundIdentifier)
            .join(Registry)
            .filter(CompoundIdentifier.compound_id == compound.id)
            .filter(CompoundIdentifier.registry_id == Registry.id)
            .filter(Registry.name == "Synonyms")
        )

        names = set()
        for identifier in query:
            names.update(identifier.accession.split("|"))
        return names

    def search(
        self, query: str, page: int = 1, page_size: int = 10
    ) -> List[Tuple[Compound, float]]:
        """Search for compounds with names similar to the query.

        Parameters
        ----------
        query : str
            the desired search term
        page : int, optional
            the number of compounds per page
            (Default value = 1)
        page_size : int, optional
            the page size for the results
            (Default value = 10)

        Returns
        -------
        List[Tuple[Compound, float]]
            A list of pairs of compounds and their scores from the
            specified search result page.
        """
        levenshtein_ratio = partial(ratio, query.lower())
        logger.debug("Scoring all synonyms with the query.")
        scores = self._synonyms["accession"].map(levenshtein_ratio)
        # Only consider the highest score for each compound.
        result = scores.groupby(scores.index, sort=False).max()
        # Filter out low scores and sort in descending order.
        hits: pd.Series = result[result > 0.5].sort_values(ascending=False)
        if len(hits) == 0:
            raise ValueError(
                f"There are no significant results for your query '{query}'."
            )
        begin = (page - 1) * page_size
        logger.debug("Collecting results from the database.")
        return [
            (self.get_compound_by_internal_id(i), s)
            for i, s in hits.iloc[begin : begin + page_size].items()
        ]

    def get_compounds(
        self,
        namespace: Optional[str],
        identifiers: Iterable[str],
        is_inchi_key: bool = False,
        is_inchi: bool = False,
    ) -> List[Compound]:
        """Retrieve many compounds within a namespace by their identifiers.

        Parameters
        ----------
        namespace : str or None
            The namespace of the identifiers, for example,
            'metanetx.chemical'. Can be `None` in case of InChIs or InChI Keys.
        identifiers : iterable of str
            The exact identifiers of compounds of interest.
        is_inchi_key : bool, optional
            Whether the identifiers should be considered as InChI Keys (
            default `False`).
        is_inchi : bool, optional
            Whether the identifiers should be considered as InChIs (
            default `False`).

        Returns
        -------
        List[Compound]
            The list of retrieved compounds.

        """
        query = (
            self.session.query(Compound)
            .outerjoin(CompoundIdentifier)
            .outerjoin(CompoundMicrospecies)
            .outerjoin(Registry)
        )
        if is_inchi_key:
            return query.filter(Compound.inchi_key.in_(identifiers)).all()
        elif is_inchi:
            return query.filter(Compound.inchi.in_(identifiers)).all()
        else:
            return query.filter(
                Registry.namespace == namespace,
                CompoundIdentifier.accession.in_(identifiers),
            ).all()
