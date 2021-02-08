"""a basic stoichiometric model with thermodynamics."""
# The MIT License (MIT)
#
# Copyright (c) 2013 Weizmann Institute of Science
# Copyright (c) 2018-2020 Institute for Molecular Systems Biology,
# ETH Zurich
# Copyright (c) 2018-2020 Novo Nordisk Foundation Center for Biosustainability,
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
import sys
import warnings
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from equilibrator_cache import Compound
from slugify import slugify

from equilibrator_api import Q_, ComponentContribution, R, Reaction, default_T

from . import Bounds, SBtabDocument, SBtabError, SBtabTable, open_sbtabdoc


class StoichiometricModel(object):
    """A basic stoichiometric model with thermodynamics.

    Designed as a base model for 'Pathway' which also includes flux directions
    and magnitudes.
    """

    CONFIDENCE_INTERVAL = 1.96  # TODO: move to params dict
    MINIMAL_STDEV = 1e-3  # TODO: move to params dict

    def __init__(
        self,
        reactions: List[Reaction],
        comp_contrib: Optional[ComponentContribution] = None,
        standard_dg_primes: Optional[Q_] = None,
        dg_sigma: Optional[Q_] = None,
        bounds: Optional[Bounds] = None,
        config_dict: Optional[Dict[str, str]] = None,
        compound_id_mapping: Callable[[Compound], str] = None,
    ) -> None:
        """Initialize a StoichiometricModel object.

        Parameters
        ----------
        reactions : List[Reaction]
            a list of Reaction objects
        comp_contrib : ComponentContribution
            a ComponentContribution object
        standard_dg_primes : Quantity, optional
            reaction energies (in kJ/mol)
        dg_sigma : Quantity, optional
            square root of the uncertainty covariance matrix (in kJ/mol)
        bounds : Bounds, optional
            bounds on metabolite concentrations (by default uses the
            "data/cofactors.csv" file in `equilibrator-api`)
        config_dict : dict, optional
            configuration parameters for Pathway analysis
        compound_id_mapping : callable, optional
            a function mapping compounds to their names in the model
        """
        self.comp_contrib = comp_contrib or ComponentContribution()
        self.water = self.comp_contrib.ccache.water
        self.is_proton = self.comp_contrib.ccache.is_proton
        self.is_water = self.comp_contrib.ccache.is_water
        self.reactions = reactions

        if bounds is None:
            self._bounds = Bounds.get_default_bounds(self.comp_contrib).copy()
        else:
            self._bounds = bounds.copy()

        self.config_dict = config_dict or dict()
        self.stdev_factor = 0.0
        self.configure()

        # make sure the bounds on Water are (1, 1)
        self._bounds.set_bounds(self.water, Q_(1.0, "M"), Q_(1.0, "M"))

        self.S = self.comp_contrib.create_stoichiometric_matrix(reactions)

        # we keep the index to H2O handy, since we often need to remove
        # water from the list of compounds
        self.idx_water = self.S.index.tolist().index(self.water)
        self.Nc, self.Nr = self.S.shape

        if standard_dg_primes is None:
            assert dg_sigma is None, (
                "If standard_dg_primes are not "
                "provided, dg_sigma must also be None"
            )
            self.update_standard_dgs()
        else:
            assert standard_dg_primes.check("[energy]/[substance]")
            assert standard_dg_primes.shape == (self.Nr,)
            if dg_sigma is not None:
                assert dg_sigma.check("[energy]/[substance]")
                # note that the number of columns in the sqrt
                # of the covariance can be any number, so we don't check it.
                assert dg_sigma.shape[0] == self.Nr

            # dGr should be orthogonal to nullspace of S
            # If not, dGr is not contained in image(S) and then there
            # is no consistent set of dGfs that generates dGr and the
            # first law of thermo is violated by the model.
            S_T = self.S.T.values
            S_inv = np.linalg.pinv(S_T)
            null_proj = np.eye(self.S.shape[1]) - S_T @ S_inv
            projected = null_proj @ standard_dg_primes.T
            assert (projected < Q_("1e-8 kJ/mol")).all(), (
                "Supplied reaction standard deltaG values are inconsistent "
                "with the stoichiometric matrix."
            )
            self.standard_dg_primes = standard_dg_primes
            self.dg_sigma = dg_sigma

        # Set the compound names, these names can
        # be changed later using the same function (set_compound_ids)
        self.set_compound_ids(compound_id_mapping)

        # Some analyses use Z-score to describe thee distribution of metabolites
        # so we need to convert the "hard" bounds into a Gaussian distribution
        # we assume that the given bounds represent the 95% confidence interval
        # a the Gaussian distribution (i.e. [mu - 1.96*sigma, mu + 1.96*sigma])
        # Therefore:
        #              mu = (ub + lb) / 2
        #           sigma = (ub - lb) / 3.92
        self.ln_conc_mu = (self.ln_conc_ub + self.ln_conc_lb) / 2.0
        self.ln_conc_sigma = (self.ln_conc_ub - self.ln_conc_lb) / (
            self.CONFIDENCE_INTERVAL * 2.0
        )

    def configure(self) -> None:
        """Configure the Component Contribution aqueous conditions."""
        if "p_h" in self.config_dict:
            self.comp_contrib.p_h = Q_(self.config_dict["p_h"])

        if "p_mg" in self.config_dict:
            self.comp_contrib.p_mg = Q_(self.config_dict["p_mg"])

        if "ionic_strength" in self.config_dict:
            self.comp_contrib.ionic_strength = Q_(
                self.config_dict["ionic_strength"]
            )

        if "temperature" in self.config_dict:
            self.comp_contrib.temperature = Q_(self.config_dict["temperature"])

        # this is the factor by which the STDEV matrix is multiplied to create
        # the confidence margins for the Gibbs energies of reaction
        self.stdev_factor = float(self.config_dict.get("stdev_factor", "0"))

    def update_standard_dgs(self) -> None:
        """Calculate the standard ΔG' values and uncertainties.

        Use the Component Contribution method.
        """
        (
            self.standard_dg_primes,
            self.dg_sigma,
        ) = self.comp_contrib.standard_dg_prime_multi(
            self.reactions, uncertainty_representation="sqrt"
        )

    def set_compound_ids(
        self, mapping: Callable[[Compound], str] = None
    ) -> None:
        """Use alternative compound names for outputs such as plots.

        Parameters
        ----------
        mapping : callable, optional
            a function mapping compounds to their names in the model

        """
        if mapping is None:
            self.compound_ids = [
                slugify(cpd.get_common_name(), separator="_", lowercase=False)
                for cpd in self.S.index
            ]
        else:
            self.compound_ids = list(map(mapping, self.S.index))

    @property
    def bounds(self) -> Tuple[Iterable[float], Iterable[float]]:
        """Get the concentration bounds.

        The order of compounds is according to the stoichiometric matrix index.

        Returns
        -------
        tuple of (lower bounds, upper bounds)
        """
        return self._bounds.get_bounds(self.S.index)

    @property
    def ln_conc_lb(self) -> np.array:
        """Get the log lower bounds on the concentrations.

        The order of compounds is according to the stoichiometric matrix index.

        Returns
        -------
        a NumPy array of the log lower bounds
        """
        return np.array(
            list(map(float, self._bounds.get_ln_lower_bounds(self.S.index)))
        )

    @property
    def ln_conc_ub(self) -> np.ndarray:
        """Get the log upper bounds on the concentrations.

        The order of compounds is according to the stoichiometric matrix index.

        Returns
        -------
        a NumPy array of the log upper bounds
        """
        return np.array(
            list(map(float, self._bounds.get_ln_upper_bounds(self.S.index)))
        )

    @staticmethod
    def get_compounds(reactions: Iterable[Reaction]) -> List[Compound]:
        """Get a unique list of all compounds in all reactions.

        :param reactions: an iterator of reactions
        :return: a list of unique compounds
        """
        compounds = set()
        for r in reactions:
            compounds.update(r.keys())

        return sorted(compounds)

    @property
    def reaction_ids(self) -> Iterable[str]:
        """Iterate through all the reaction IDs.

        :return: the reaction IDs
        """
        return map(lambda rxn: rxn.rid, self.reactions)

    @property
    def reaction_formulas(self) -> Iterable[str]:
        """Iterate through all the reaction formulas.

        :return: the reaction formulas
        """
        return map(self._reaction_to_formula, self.reactions)

    @staticmethod
    def read_thermodynamics(
        thermo_sbtab: SBtabTable, config_dict: Dict[str, str]
    ) -> Dict[str, Q_]:
        """Read the 'thermodynamics' table from an SBtab.

        Parameters
        ----------
        thermo_sbtab : SBtabTable
            A SBtabTable containing the thermodynamic data
        config_dict : dict
            A dictionary containing the configuration arguments

        Returns
        -------
        A dictionary mapping reaction IDs to standard ΔG' values.
        """
        try:
            std_conc = thermo_sbtab.get_attribute("StandardConcentration")
            assert Q_(std_conc) == Q_(
                "1 M"
            ), "We only support a standard concentration of 1 M."
        except SBtabError:
            pass

        if "temperature" in config_dict:
            temperature = Q_(config_dict["temperature"])
            assert temperature.check("[temperature]")
        else:
            temperature = default_T

        thermo_df = thermo_sbtab.to_data_frame()
        cols = ["QuantityType", "Value", "Compound", "Reaction", "Unit"]

        rid2dg0 = dict()
        for i, row in thermo_df.iterrows():
            try:
                typ, val, cid, rid, unit = [row[c] for c in cols]
                val = Q_(float(val), unit)

                if typ == "equilibrium constant":
                    assert val.check(
                        ""
                    ), "equilibrium constants must have no units"
                    rid2dg0[rid] = -np.log(val.m_as("")) * R * temperature
                elif typ == "reaction gibbs energy":
                    assert val.check(
                        "[energy]/[substance]"
                    ), "Gibbs energies must be in kJ/mol (or equivalent)"
                    val.ito("kJ/mol")
                    rid2dg0[rid] = val
                else:
                    raise AssertionError(
                        "unrecognized Rate Constant Type: " + typ
                    )
            except AssertionError:
                raise ValueError(
                    "Syntax error in Parameter table, row %d - %s" % (i, row)
                )
        return rid2dg0

    def _compound_to_id(self, compound: Compound) -> str:
        d = dict(zip(self.S.index, self.compound_ids))
        try:
            return d[compound]
        except KeyError:
            warnings.warn(
                f"Cannot find {compound.get_common_name()} in the model"
            )
            return compound.get_common_name()

    def _write_compound_and_coeff(
        self, compound: Compound, coeff: float
    ) -> str:
        if np.abs(coeff - 1.0) < sys.float_info.epsilon:
            return self._compound_to_id(compound)
        else:
            return "%g %s" % (coeff, self._compound_to_id(compound))

    def _reaction_to_formula(self, reaction: Reaction) -> str:
        l_tokens = []
        r_tokens = []
        for compound, coeff in sorted(reaction.sparse.items()):
            if self.is_proton(compound):
                continue
            elif coeff < 0:
                l_tokens.append(
                    self._write_compound_and_coeff(compound, -coeff)
                )
            elif coeff > 0:
                r_tokens.append(self._write_compound_and_coeff(compound, coeff))

        l_str = " + ".join(l_tokens)
        r_str = " + ".join(r_tokens)
        return f"{l_str} {reaction.arrow} {r_str}"

    @staticmethod
    def _read_network_sbtab(
        filename: Union[str, SBtabDocument],
        comp_contrib: Optional[ComponentContribution] = None,
        freetext: bool = True,
    ) -> Tuple[pd.DataFrame, ComponentContribution, Dict[str, str]]:
        """Initialize a Pathway object using a 'network'-only SBtab.

        Parameters
        ----------
        filename : str, SBtabDocument
             a filename containing an SBtabDocument (or the SBtabDocument
             object itself) defining the network (topology) only
        comp_contrib : ComponentContribution, optional
             a ComponentContribution object needed for parsing and searching
             the reactions.
             also used to set the aqueous parameters (pH, I, etc.)
        freetext : bool, optional
             a flag indicating whether the reactions are given as free-text
             (i.e. common names for compounds) or by standard database
             accessions. (Default value: `True`)

        Returns
        -------
        reaction_df
        comp_contrib
        config_dict
        """

        def formula_to_obj(rxn_formula: str) -> Reaction:
            """Convert the reaction formulas to Reaction objects."""
            logging.debug("formula = %f", rxn_formula)

            if freetext:
                rxn = comp_contrib.search_reaction(rxn_formula)
            else:
                rxn = comp_contrib.parse_reaction_formula(rxn_formula)
            if not rxn.is_balanced():
                raise Exception(f"This reaction is not balanced: {rxn_formula}")
            return rxn

        comp_contrib = comp_contrib or ComponentContribution()

        sbtabdoc = open_sbtabdoc(filename)

        reaction_df = sbtabdoc.get_sbtab_by_id("Reaction").to_data_frame()
        reaction_df["Reaction"] = reaction_df.ReactionFormula.apply(
            formula_to_obj
        )
        for i, rxn in enumerate(reaction_df["Reaction"]):
            rxn.rid = f"R{i:05d}"

        config_dict = {
            "p_h": str(comp_contrib.p_h),
            "p_mg": str(comp_contrib.p_mg),
            "ionic_strength": str(comp_contrib.ionic_strength),
            "temperature": str(comp_contrib.temperature),
        }
        return reaction_df, comp_contrib, config_dict

    @staticmethod
    def _read_model_sbtab(
        filename: Union[str, SBtabDocument],
        comp_contrib: Optional[ComponentContribution] = None,
    ):
        """Parse and SBtabDocument and return a StoichiometricModel.

        Parameters
        ----------
        filename : str or SBtabDocument
            a filename containing an SBtabDocument (or the SBtabDocument
            object itself) defining the pathway
        comp_contrib : ComponentContribution, optional
            a ComponentContribution object needed for parsing and searching
            the reactions. also used to set the aqueous parameters (pH, I, etc.)

        Returns
        -------
        stoich_model: StoichiometricModel
            A StoichiometricModel object based on the configuration SBtab

        """
        sbtabdoc = open_sbtabdoc(filename)

        # Read the configuration table if it exists
        config_sbtab = sbtabdoc.get_sbtab_by_id("Configuration")
        if config_sbtab:
            config_df = config_sbtab.to_data_frame()
            assert (
                "Option" in config_df.columns
            ), "Configuration table must have an Option column"
            assert (
                "Value" in config_df.columns
            ), "Configuration table must have an Value column"
            config_dict = config_df.set_index("Option").Value.to_dict()
        else:
            config_dict = dict()

        table_ids = ["Compound", "ConcentrationConstraint", "Reaction"]
        dfs = []

        for table_id in table_ids:
            sbtab = sbtabdoc.get_sbtab_by_id(table_id)
            if sbtab is None:
                tables = ", ".join(map(lambda s: s.table_id, sbtabdoc.sbtabs))
                raise ValueError(
                    f"The SBtabDocument must have a table "
                    f"with the following ID: {table_id}, "
                    f"however, only these tables were "
                    f"found: {tables}"
                )
            dfs.append(sbtab.to_data_frame())

        compound_df, bounds_df, reaction_df = dfs

        # Read the Compound table
        # -----------------------
        # use equilibrator-cache to build a dictionary of compound IDs to
        # Compound objects
        compound_df.set_index("ID", inplace=True)

        # legacy option, KEGG IDs could be provided without the namespace
        # when the column title  explicitly indicated that they were from KEGG
        if "Identifiers:kegg.compound" in compound_df.columns:
            compound_df["Identifiers"] = (
                "kegg:" + compound_df["Identifiers:kegg.compound"]
            )

        # Now, by default, we assume the namespaces are provided as part of the
        # reaction formulas
        if "Identifiers" not in compound_df.columns:
            raise KeyError(
                "There is no column of Identifiers in the Compound table"
            )

        comp_contrib = comp_contrib or ComponentContribution()

        compounds = []
        for i, row in enumerate(compound_df.itertuples()):
            if row.Identifiers:
                compounds.append(comp_contrib.get_compound(row.Identifiers))
            else:
                # generate a novel Compound object (with a unique negative ID)
                # just to use it later in this specific pathway
                compounds.append(Compound(id=-i))
        compound_df["Compound"] = compounds

        if pd.isnull(compound_df.Compound).any():
            accessions_not_found = compound_df.loc[
                pd.isnull(compound_df.Compound), "Identifiers"
            ]
            error_msg = str(accessions_not_found.to_dict())
            raise KeyError(
                f"Some compounds not found in equilibrator-cache: "
                f"{error_msg}"
            )

        # Read the ConcentrationConstraints table
        # ---------------------------------------
        # convert compound IDs in the bounds table to Compound objects
        id_to_compound = compound_df.Compound.to_dict()
        bounds_df["Compound"] = bounds_df.Compound.apply(id_to_compound.get)
        bounds_df.set_index("Compound", inplace=True)

        bounds_sbtab = sbtabdoc.get_sbtab_by_id("ConcentrationConstraint")
        try:
            bounds_unit = bounds_sbtab.get_attribute("Unit")
            lbs = bounds_df["Min"].apply(lambda x: Q_(float(x), bounds_unit))
            ubs = bounds_df["Max"].apply(lambda x: Q_(float(x), bounds_unit))
        except SBtabError:
            # if the unit is not defined in the header, we assume it is given
            # next to each bound individually
            lbs = bounds_df["Min"].apply(Q_)
            ubs = bounds_df["Max"].apply(Q_)

        bounds = Bounds(lbs.to_dict(), ubs.to_dict())
        bounds.check_bounds()

        # Read the Reaction table
        # -----------------------
        reactions = []
        reaction_ids = []
        for row in reaction_df.itertuples():
            rxn = Reaction.parse_formula(
                id_to_compound.get, row.ReactionFormula
            )
            rxn.rid = row.ID
            if not rxn.is_balanced(ignore_atoms=("H",)):
                warnings.warn(f"Reaction {rxn.rid} is not balanced")
            reactions.append(rxn)
            if row.ID in reaction_ids:
                raise KeyError(
                    f"Reaction IDs must be unique, but you have "
                    f"{row.ID} twice"
                )
            reaction_ids.append(row.ID)

        thermo_sbtab = sbtabdoc.get_sbtab_by_id("Thermodynamics")
        if thermo_sbtab:
            rid2dg0 = StoichiometricModel.read_thermodynamics(
                thermo_sbtab, config_dict
            )

            # make sure all reactions have a standard Gibbs energy
            assert set(reaction_ids).issubset(rid2dg0.keys()), (
                "Not all reactions have a equilibrium constant or gibbs energy "
                "in the 'Thermodynamics' table."
            )
            standard_dg_primes = np.array(
                [rid2dg0[rid].m_as("kJ/mol") for rid in reaction_ids]
            ) * Q_("1 kJ/mol")
            dg_sigma = None
        else:
            standard_dg_primes = None
            dg_sigma = None

        compound_to_id = dict(map(reversed, id_to_compound.items()))
        return (
            sbtabdoc,
            reactions,
            reaction_ids,
            standard_dg_primes,
            dg_sigma,
            bounds,
            config_dict,
            compound_to_id,
        )

    @classmethod
    def from_network_sbtab(
        cls,
        filename: Union[str, SBtabDocument],
        comp_contrib: Optional[ComponentContribution] = None,
        freetext: bool = True,
        bounds: Optional[Bounds] = None,
    ) -> object:
        """Initialize a Pathway object using a 'network'-only SBtab.

        Parameters
        ----------
        filename : str, SBtabDocument
            a filename containing an SBtabDocument (or the SBtabDocument
            object itself) defining the network (topology) only
        comp_contrib : ComponentContribution, optional
            a ComponentContribution object needed for parsing and searching
            the reactions. also used to set the aqueous parameters (pH, I, etc.)
        freetext : bool, optional
            a flag indicating whether the reactions are given as free-text (i.e.
            common names for compounds) or by standard database accessions
            (Default value: `True`)
        bounds : Bounds, optional
            bounds on metabolite concentrations (by default uses the
            "data/cofactors.csv" file in `equilibrator-api`)

        Returns
        -------
            a Pathway object
        """
        (
            reaction_df,
            comp_contrib,
            config_dict,
        ) = StoichiometricModel._read_network_sbtab(
            filename, comp_contrib, freetext
        )
        stoich_model = StoichiometricModel(
            reactions=reaction_df.Reaction.tolist(),
            comp_contrib=comp_contrib,
            bounds=bounds,
            config_dict=config_dict,
        )
        return stoich_model

    @classmethod
    def from_sbtab(
        cls,
        filename: Union[str, SBtabDocument],
        comp_contrib: Optional[ComponentContribution] = None,
    ) -> "StoichiometricModel":
        """Parse and SBtabDocument and return a StoichiometricModel.

        Parameters
        ----------
        filename : str or SBtabDocument
            a filename containing an SBtabDocument (or the SBtabDocument
            object itself) defining the pathway
        comp_contrib : ComponentContribution, optional
            a ComponentContribution object needed for parsing and searching
            the reactions. also used to set the aqueous parameters (pH, I, etc.)

        Returns
        -------
        stoich_model: StoichiometricModel
            A StoichiometricModel object based on the configuration SBtab

        """
        (
            sbtabdoc,
            reactions,
            reaction_ids,
            standard_dg_primes,
            dg_sigma,
            bounds,
            config_dict,
            compound_to_id,
        ) = cls._read_model_sbtab(filename, comp_contrib)

        return StoichiometricModel(
            reactions=reactions,
            comp_contrib=comp_contrib,
            standard_dg_primes=standard_dg_primes,
            dg_sigma=None,
            bounds=bounds,
            config_dict=config_dict,
            compound_id_mapping=compound_to_id.get,
        )

    def to_sbtab(self) -> SBtabDocument:
        """Export the model to an SBtabDocument."""
        sbtabdoc = SBtabDocument("pathway", filename="pathway.tsv")

        config_dict = dict(self.config_dict)
        config_dict["algorithm"] = "MDF"
        config_dict["stdev_factor"] = "0.0"
        config_df = pd.DataFrame(
            data=[(k, v, "") for k, v in config_dict.items()],
            columns=["!Option", "!Value", "!Comment"],
        )
        config_sbtab = SBtabTable.from_data_frame(
            df=config_df, table_id="Configuration", table_type="Config"
        )

        reaction_df = pd.DataFrame(
            data=[
                (rxn.rid, self._reaction_to_formula(rxn))
                for rxn in self.reactions
            ],
            columns=["!ID", "!ReactionFormula"],
        )
        reaction_sbtab = SBtabTable.from_data_frame(
            df=reaction_df, table_id="Reaction", table_type="Reaction"
        )

        compound_df = pd.DataFrame(
            data=[
                (
                    self._compound_to_id(cpd),
                    cpd.get_common_name(),
                    cpd.get_accession(),
                )
                for cpd in self.S.index
            ],
            columns=["!ID", "!Name", "!Identifiers"],
        )
        compound_sbtab = SBtabTable.from_data_frame(
            df=compound_df, table_id="Compound", table_type="Compound"
        )

        thermo_df = pd.DataFrame(
            data=[
                (
                    "reaction gibbs energy",
                    rxn.rid,
                    "",
                    f"{dg.m_as('kJ/mol'):.2f}",
                    "kJ/mol",
                )
                for rxn, dg in zip(self.reactions, self.standard_dg_primes)
            ],
            columns=[
                "!QuantityType",
                "!Reaction",
                "!Compound",
                "!Value",
                "!Unit",
            ],
        )
        thermo_sbtab = SBtabTable.from_data_frame(
            df=thermo_df, table_id="Thermodynamics", table_type="Quantity"
        )
        thermo_sbtab.change_attribute("StandardConcentration", "M")

        lbs, ubs = self.bounds
        lbs = map(lambda x: x.m_as("mM"), lbs)
        ubs = map(lambda x: x.m_as("mM"), ubs)
        conc_df = pd.DataFrame(
            data=[
                (
                    "concentration",
                    self._compound_to_id(cpd),
                    f"{lb:.3g}",
                    f"{ub:.3g}",
                )
                for cpd, lb, ub in zip(self.S.index, lbs, ubs)
            ],
            columns=["!QuantityType", "!Compound", "!Min", "!Max"],
        )
        conc_sbtab = SBtabTable.from_data_frame(
            df=conc_df,
            table_id="ConcentrationConstraint",
            table_type="Quantity",
        )
        conc_sbtab.change_attribute("Unit", "mM")

        sbtabdoc.add_sbtab(config_sbtab)
        sbtabdoc.add_sbtab(reaction_sbtab)
        sbtabdoc.add_sbtab(compound_sbtab)
        sbtabdoc.add_sbtab(thermo_sbtab)
        sbtabdoc.add_sbtab(conc_sbtab)
        return sbtabdoc

    def write_sbtab(self, filename: str) -> None:
        """Write the pathway to an SBtab file."""
        self.to_sbtab().write(filename)
