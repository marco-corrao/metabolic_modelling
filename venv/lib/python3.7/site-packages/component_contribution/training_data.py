# The MIT License (MIT)
#
# Copyright (c) 2013 The Weizmann Institute of Science.
# Copyright (c) 2018 Novo Nordisk Foundation Center for Biosustainability,
# Technical University of Denmark.
# Copyright (c) 2018 Institute for Molecular Systems Biology,
# ETH Zurich, Switzerland.
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


"""
Manage the training data for component contributions.

References:
-----------
.. [1] Alberty (2006)
.. [2] Maden (2000)
.. [3] Thauer (1977)
.. [4] Wagman (1982)
.. [5] Dolfing (1992)
.. [6] Dolfing (1994)
.. [7] CRC biochemistry (2010)
.. [8] Prince (1987)
.. [9] Thauer (1977)
.. [10] CRC biochemistry (2010)
.. [11] Alberty (2006)
.. [12] Deppenmeier (2008)
.. [13] Saeki (1985)
.. [14] Unden (1997)

"""
import logging
import warnings
from io import BytesIO
from typing import Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import pkg_resources
from equilibrator_cache import FARADAY, Q_, Compound, CompoundCache, R, Reaction
from equilibrator_cache.exceptions import (
    MissingDissociationConstantsException,
    ParseException,
)
from equilibrator_cache.reaction import (
    create_stoichiometric_matrix_from_reactions,
)

from . import ZENODO_DOI_GROUPS, ZENODO_DOI_TRAINING_DATA
from .diskcache import get_cached_file


logger = logging.getLogger(__name__)


class TrainingData(object):
    """A base class for handling Training Data."""

    def __init__(
        self,
        ccache: CompoundCache,
        override_ionic_strength: Optional[Q_] = None,
        override_temperature: Optional[Q_] = None,
        override_p_mg: Optional[Q_] = None,
    ):
        """
        Create a TrainingData object.

        Parameters
        ----------
        ccache : CompoundCache
            The compound cache used for looking up structures and pKas.
        override_ionic_strength : bool, optional
            If provided, overrides all ionic_strength values given in the data
            files with this value.
        override_temperature : bool, optional
            If provided, overrides all temperature values given in the data
            files with this value.
        override_p_mg : bool, optional
            If provided, overrides all p_mg values given in the data
            files with this value.

        """
        self.ccache = ccache
        self.S = None  # a DataFrame containing the stoichiometric matrix

        # read the group summary table from Zenodo
        group_file = get_cached_file(ZENODO_DOI_GROUPS, "group_summary.csv")
        self.group_summary = pd.read_csv(group_file, index_col=0)

        logger.info("Gathering reaction data")
        (
            self.reaction_df,
            self._non_decomposable_compounds,
        ) = self.create_reaction_dataframe()

        for cpd in self._non_decomposable_compounds:
            if self.ccache.is_proton(cpd):
                raise ValueError("Protons are not allowed in the training data")

        for compound in self.decomposable_compounds:
            if compound.atom_bag is None and compound.inchi is None:
                warnings.warn(
                    f"Compound {compound} has neither an atom "
                    f"bag nor an InChI."
                )
            elif compound.atom_bag is None and compound.inchi is not None:
                warnings.warn(
                    f"Compound {compound} has no atom bag, but "
                    f"does have an InChI: {compound.inchi}."
                )

        logger.info("Ensure all the relevant columns are in the correct units")
        self.assert_data()

        logger.info("Balancing reactions with H2O and H+")
        self.balance_reactions()

        logger.info("Creating the training stoichiometric matrix")
        self.create_stoichiometric_matrix_from_reactions()

        logger.info("Applying reverse Legendre transform on dG' values")
        standard_dg = self._reverse_transform(
            override_ionic_strength=override_ionic_strength,
            override_temperature=override_temperature,
            override_p_mg=override_p_mg,
        )
        self.reaction_df = self.reaction_df.assign(
            **{"standard_dg": list(standard_dg)}
        )

    def parse_formula(self, formula: str) -> Reaction:
        """Parse a chemical formula to create a Reaction object using ccache."""
        return Reaction.parse_formula(self.ccache.get_compound, formula)

    def create_reaction_dataframe(self) -> Tuple[pd.DataFrame, Set[Compound]]:
        """Create the DataFrame containing all the reaction training data."""
        raise NotImplementedError()

    @property
    def stoichiometric_matrix(self) -> pd.DataFrame:
        """Get the stoichiometric matrix as a DataFrame.

        :return: a Pandas DataFrame
        """
        return self.S

    @property
    def compounds(self) -> List[Compound]:
        """Get the list of compounds.

        :return: A list of Compound objects according to their order in S.
        """
        if self.S is None:
            compounds = {
                c
                for r in self.reaction_df.reaction
                for c in r.sparse
                if not self.ccache.is_proton(c)
            }
            if not any(map(self.ccache.is_water, compounds)):
                compounds.add(self.ccache.water)
            return sorted(compounds)
        else:
            return self.S.index.tolist()

    @property
    def non_decomposable_compounds(self) -> Set[Compound]:
        """Get the set of non-decomposable compounds.

        :return: A Set of Compound objects
        """
        return self._non_decomposable_compounds

    @property
    def decomposable_compounds(self) -> Set[Compound]:
        """Get the set of decomposable compounds.

        :return: A Set of Compound objects
        """
        return set(self.compounds).difference(self._non_decomposable_compounds)

    @property
    def standard_dg(self) -> pd.Series:
        """Get the standard deltaGs according to the reaction order in S.

        :return: A Pandas Series of the standard detlaGs
        """
        return self.reaction_df.standard_dg

    @property
    def weight(self) -> pd.Series:
        """Get the weights according to the reaction order in S.

        :return:A Pandas Series of the weights
        """
        return self.reaction_df.weight

    def assert_data(self):
        """Ensure all the relevant columns are in the correct units."""
        for col, dim in [
            ("standard_dg_prime", "[energy]/[substance]"),
            ("ionic_strength", "[concentration]"),
            ("temperature", "[temperature]"),
            ("p_h", None),
            ("p_mg", None),
        ]:
            assert col in self.reaction_df.columns

            for v in self.reaction_df[col].values:
                assert v.check(dim)

    def _balance_reaction(self, rxn: Reaction) -> Union[Reaction, None]:
        """Try balancing the reaction with H2O and H+.

        :param rxn: A Reaction object
        :return: A balanced Reaction, or None
        """
        if rxn.is_balanced(ignore_atoms=()):
            # the reaction is already balanced, nothing to do
            return rxn

        if rxn.is_balanced(ignore_atoms=("H",)):
            # the reaction just needs to be balanced by protons
            return rxn.balance_with_compound(
                self.ccache.proton, ignore_atoms=()
            )

        if not rxn.is_balanced(ignore_atoms=("H", "O", "e-")):
            # this reaction cannot be balanced only by H2O and H+, because
            # elements other than e-, O, and H are not balanced
            warnings.warn("This reaction cannot be balanced: " f"{rxn}")
            return None
        else:
            # we need to first balance the O atoms using water, and then
            # balance the H atoms using protons
            rxn = rxn.balance_with_compound(
                self.ccache.water, ignore_atoms=("H",)
            )
            return rxn.balance_with_compound(
                self.ccache.proton, ignore_atoms=()
            )

    def balance_reactions(self) -> None:
        """Balance the reactions.

        Use the chemical formulas from the InChIs to verify that each and
        every reaction is balanced. If it can be easily fixed with protons
        and/or water, fix it. Otherwise, drop this reaction from the training
        data.
        """
        balanced_reactions = []
        for row in self.reaction_df.itertuples(index=True):
            if "balance" in self.reaction_df.columns and not row.balance:
                balanced_reactions.append(row.reaction)
            else:
                balanced_reactions.append(self._balance_reaction(row.reaction))

        self.reaction_df["reaction"] = balanced_reactions
        self.reaction_df = self.reaction_df[
            ~pd.isnull(self.reaction_df.reaction)
        ]

    def create_stoichiometric_matrix_from_reactions(self) -> None:
        """Create the stoichiometric matrix for training.

        Convert the list of reactions in sparse notation into a full
        stoichiometric matrix, where the rows (compounds) are in the same
        order as in 'compounds' and the columns match the reaction indices.
        """
        self.S = create_stoichiometric_matrix_from_reactions(
            self.reaction_df.reaction,
            self.ccache.is_proton,
            self.ccache.is_water,
            self.ccache.water,
        )
        self.S.columns = self.reaction_df.index

    def _reverse_transform(
        self,
        override_ionic_strength: Optional[Q_] = None,
        override_temperature: Optional[Q_] = None,
        override_p_mg: Optional[Q_] = None,
    ) -> Iterable[float]:
        """Reverse Legendre transform.

        Parameters
        ----------
        override_ionic_strength : bool, optional
            If provided, overrides all ionic_strength values given in the data
            files with this value.
        override_temperature : bool, optional
            If provided, overrides all temperature values given in the data
            files with this value.
        override_p_mg : bool, optional
            If provided, overrides all p_mg values given in the data
            files with this value.

        """
        _reaction_df = self.reaction_df.copy()
        if override_ionic_strength:
            assert override_ionic_strength.check("[concentration]")
            _reaction_df.ionic_strength = override_ionic_strength
        if override_temperature:
            assert override_temperature.check("[temperature]")
            _reaction_df.temperature = override_temperature
        if override_p_mg:
            assert override_p_mg.check("")
            _reaction_df.p_mg = override_p_mg

        for row in _reaction_df.itertuples():
            standard_dg = row.standard_dg_prime
            try:
                standard_dg -= row.reaction.transform(
                    p_h=row.p_h,
                    ionic_strength=row.ionic_strength,
                    temperature=row.temperature,
                    p_mg=row.p_mg,
                )
            except MissingDissociationConstantsException as e:
                logger.warning(f"Cannot reverse transform {row.reaction}: {e}")
            yield standard_dg


class ToyTrainingData(TrainingData):
    """Toy training data class."""

    def __init__(self, **kw):
        """Create the ToyTrainingData object."""
        super(ToyTrainingData, self).__init__(**kw)

    def create_reaction_dataframe(self) -> Tuple[pd.DataFrame, Set[Compound]]:
        """Create the DataFrame of reactions for training."""
        reaction_df = pd.read_csv(
            pkg_resources.resource_filename(
                "component_contribution.data", "toy_training_data.csv"
            )
        )

        # convert the formula strings into Reaction objects
        reaction_df.loc[:, "reaction"] = reaction_df.reaction.apply(
            self.parse_formula
        )

        # add units to the relevant columns:
        for col, unit in [
            ("standard_dg_prime", "kJ/mol"),
            ("ionic_strength", "M"),
            ("temperature", "K"),
            ("p_h", None),
            ("p_mg", None),
        ]:
            reaction_df[col] = reaction_df[col].apply(lambda x: Q_(x, unit))

        non_decomposable_compounds = set()
        return reaction_df, non_decomposable_compounds


class FullTrainingData(TrainingData):
    """Full training data class."""

    def __init__(self, ccache: CompoundCache, **kw):
        """Create the FullTrainingData object."""
        super(FullTrainingData, self).__init__(ccache=ccache, **kw)

    def create_reaction_dataframe(self) -> Tuple[pd.DataFrame, Set[Compound]]:
        """Create the DataFrame of reactions for training."""
        tecr_df = self.read_tecrdb(
            get_cached_file(ZENODO_DOI_TRAINING_DATA, "TECRDB.csv")
        )
        tecr_df["weight"] = 1.0

        formation_df, non_decomposable_compounds = self.read_formations(
            get_cached_file(
                ZENODO_DOI_TRAINING_DATA, "formation_energies_transformed.csv"
            )
        )
        formation_df["weight"] = 1.0

        redox_df, more_non_decomposable_compounds = self.read_redox(
            get_cached_file(ZENODO_DOI_TRAINING_DATA, "redox.csv")
        )
        redox_df["weight"] = 1.0
        non_decomposable_compounds.update(more_non_decomposable_compounds)

        reaction_df = pd.concat([tecr_df, formation_df, redox_df], sort=False)
        reaction_df.reset_index(drop=True, inplace=True)

        return reaction_df, non_decomposable_compounds

    def read_tecrdb(self, tecr_file: BytesIO) -> pd.DataFrame:
        """Load a data frame with information from the TECRdb (NIST).

        The component-contribution package distributes data tables with
        information on the 'thermodynamics of enzyme-catalyzed
        reactions'[1, 2]_ that are used as training data.

        :return: a Pandas DataFrame of the TECRDB reaction

        References
        ----------
        .. [1] Goldberg, Robert N., Yadu B. Tewari, and Talapady N. Bhat.
               “Thermodynamics of Enzyme-Catalyzed Reactions—a Database for
               Quantitative Biochemistry.” Bioinformatics 20, no. 16
               (November 1, 2004): 2874–77.
               https://doi.org/10.1093/bioinformatics/bth314.
        .. [2] http://xpdb.nist.gov/enzyme_thermodynamics/

        """
        tecr_df = pd.read_csv(tecr_file)

        # assume a default ionic strength of 0.25 M
        tecr_df.ionic_strength.fillna(0.25, inplace=True)

        # assume a default pMg of 14 (i.e. a negligible [Mg2+])
        tecr_df.p_mg.fillna(14, inplace=True)

        # remove rows with missing essential data
        tecr_df = tecr_df[
            ~pd.isnull(tecr_df[["K_prime", "temperature", "p_h"]]).any(axis=1)
        ]

        for col, unit in [
            ("K_prime", None),
            ("ionic_strength", "M"),
            ("temperature", "K"),
            ("p_h", None),
            ("p_mg", None),
        ]:
            tecr_df[col] = tecr_df[col].apply(lambda x: Q_(x, unit))

        # calculate the dG'0 from the Keq and T
        standard_dg_primes = [
            (-R * row.temperature * np.log(row.K_prime)).to("kJ/mol")
            for row in tecr_df.itertuples()
        ]
        tecr_df = tecr_df.assign(
            **{"standard_dg_prime": standard_dg_primes, "balance": True}
        )

        # parse the reaction
        tecr_df.loc[:, "reaction"] = tecr_df["reaction"].apply(
            self.parse_formula
        )

        # skip reactions that could not be parsed
        tecr_df = tecr_df[~pd.isnull(tecr_df["reaction"])]

        tecr_df.drop(
            ["url", "method", "K", "K_prime", "eval", "EC", "enzyme_name"],
            axis=1,
            inplace=True,
        )

        logger.debug(
            "Successfully added %d reactions from TECRDB" % tecr_df.shape[0]
        )
        return tecr_df

    def read_formations(
        self, formation_file: BytesIO
    ) -> Tuple[pd.DataFrame, Set[Compound]]:
        """Read the Formation Energy data from literature.

        All the data we have for formation energies from [1-6].

        :return: a 2-tuple of the Pandas DataFrame of reactions, and the set
        of compounds that do not decompose into groups.
        """
        formation_df = pd.read_csv(formation_file)
        compounds = formation_df["cid"].apply(self.ccache.get_compound)

        if pd.isnull(compounds).any():
            missing_cids = formation_df.loc[pd.isnull(compounds), "cid"]
            raise ParseException(
                "Cannot find some of the compounds in the "
                "cache: " + str(missing_cids)
            )

        compounds_that_do_not_decompose = set(
            compounds.loc[formation_df.decompose == 0].values
        )

        # skip compounds that have no formation energy (they are in the table
        # only in order to indicate something, for example that they should
        # not be decomposed)
        formation_df = formation_df[~pd.isnull(formation_df.standard_dg_prime)]

        for col, unit in [
            ("standard_dg_prime", "kJ/mol"),
            ("ionic_strength", "M"),
            ("temperature", "K"),
            ("p_h", None),
            ("p_mg", None),
        ]:
            formation_df[col] = formation_df[col].apply(lambda x: Q_(x, unit))

        formation_df = formation_df.assign(
            **{
                "reaction": compounds.apply(lambda c: Reaction({c: 1})),
                "balance": False,
                "description": formation_df["name"] + " formation",
            }
        )
        formation_df.drop(
            ["name", "cid", "remark", "decompose"], axis=1, inplace=True
        )

        logger.debug(
            "Successfully added %d formation energies" % formation_df.shape[0]
        )
        return formation_df, compounds_that_do_not_decompose

    def read_redox(
        self, redox_file: BytesIO
    ) -> Tuple[pd.DataFrame, Set[Compound]]:
        """Read the Reduction potential from literature.

        All the reduction potentials we have from [7-14].

        :return: a Pandas DataFrame of the redox reactions.
        """
        redox_df = pd.read_csv(redox_file)

        for col, unit in [
            ("standard_E_prime", "V"),
            ("ionic_strength", "M"),
            ("temperature", "K"),
            ("p_h", None),
            ("p_mg", None),
        ]:
            redox_df[col] = redox_df[col].apply(lambda x: Q_(x, unit))

        compounds_ox = redox_df["CID_ox"].apply(self.ccache.get_compound)
        compounds_red = redox_df["CID_red"].apply(self.ccache.get_compound)

        compounds_that_do_not_decompose = set(
            compounds_ox.loc[redox_df.decompose == 0].values
        )
        compounds_that_do_not_decompose.update(
            compounds_red.loc[redox_df.decompose == 0].values
        )

        reaction = [
            Reaction({c_ox: -1, c_red: 1})
            for c_ox, c_red in zip(compounds_ox, compounds_red)
        ]

        delta_nH = redox_df.nH_red - redox_df.nH_ox
        delta_charge = redox_df.charge_red - redox_df.charge_ox
        redox_df["delta_e"] = delta_nH - delta_charge

        standard_dg_primes = [
            (-FARADAY * row.standard_E_prime * row.delta_e).to("kJ/mol")
            for row in redox_df.itertuples()
        ]

        redox_df = redox_df.assign(
            **{
                "reaction": reaction,
                "description": redox_df["name"] + " redox",
                "standard_dg_prime": standard_dg_primes,
                "balance": False,
            }
        )
        redox_df.drop(
            [
                "name",
                "CID_ox",
                "CID_red",
                "charge_ox",
                "charge_red",
                "nH_ox",
                "nH_red",
                "standard_E_prime",
                "delta_e",
            ],
            axis=1,
            inplace=True,
        )

        logger.debug(
            "Successfully added %d redox potentials" % redox_df.shape[0]
        )
        return redox_df, compounds_that_do_not_decompose
