"""A wrapper for the GibbeEnergyPredictor in component-contribution."""
# The MIT License (MIT)
#
# Copyright (c) 2013 Weizmann Institute of Science
# Copyright (c) 2018 Institute for Molecular Systems Biology,
# ETH Zurich
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
import warnings
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from component_contribution import (
    ZENODO_DOI_PARAMETERS,
    ZENODO_DOI_PARAMETERS_LEGACY,
)
from component_contribution.linalg import LINALG
from component_contribution.predict import (
    CCModelParameters,
    GibbsEnergyPredictor,
)
from equilibrator_cache import (
    DEFAULT_ZENODO_DOI,
    PROTON_INCHI_KEY,
    Compound,
    CompoundCache,
    CompoundMicrospecies,
    create_compound_cache_from_zenodo,
)
from equilibrator_cache.exceptions import MissingDissociationConstantsException
from equilibrator_cache.reaction import (
    create_stoichiometric_matrix_from_reactions,
)

from . import (
    FARADAY,
    Q_,
    R,
    default_physiological_ionic_strength,
    default_physiological_p_h,
    default_physiological_p_mg,
    default_physiological_temperature,
    default_rmse_inf,
    ureg,
)
from .phased_reaction import PhasedReaction
from .reaction_parser import make_reaction_parser


logger = logging.getLogger(__name__)


def find_most_abundance_ms(
    cpd: Compound, p_h: Q_, p_mg: Q_, ionic_strength: Q_, temperature: Q_
) -> CompoundMicrospecies:
    """Find the most abundant microspecies based on transformed energies."""
    ddg_over_rts = [
        (
            ms.transform(
                pH=p_h.m_as(""),
                pMg=p_mg.m_as(""),
                ionic_strength_M=ionic_strength.m_as("M"),
                T_in_K=temperature.m_as("K"),
            ),
            ms,
        )
        for ms in cpd.microspecies
    ]
    min_ddg, min_ms = min(ddg_over_rts, key=lambda x: x[0])
    return min_ms


def predict_protons_and_charge(
    rxn: PhasedReaction, p_h: Q_, p_mg: Q_, ionic_strength: Q_, temperature: Q_
) -> Tuple[float, float, float]:
    """Find the #protons and charge of a transport half-reaction."""
    n_h = 0
    n_mg = 0
    z = 0
    for cpd, coeff in rxn.items():
        if cpd.inchi_key == PROTON_INCHI_KEY:
            n_h += coeff
            z += coeff
        else:
            ms = find_most_abundance_ms(
                cpd, p_h, p_mg, ionic_strength, temperature
            )
            n_h += coeff * ms.number_protons
            n_mg += coeff * ms.number_magnesiums
            z += coeff * ms.charge
    return n_h, n_mg, z


class ComponentContribution(object):
    """A wrapper class for GibbsEnergyPredictor.

    Also holds default conditions for compounds in the different phases.
    """

    def __init__(
        self,
        rmse_inf: Q_ = default_rmse_inf,
        ccache: Optional[CompoundCache] = None,
        predictor: Optional[GibbsEnergyPredictor] = None,
    ) -> None:
        """Create a ComponentContribution object with default settings.

        Parameters
        ----------
        rmse_inf : Quantity, optional
            Set the factor by which to multiply the error
            covariance matrix for reactions outside the span of
            Component Contribution.
        ccache : CompoundCache, optional
        predictor : GibbsEnergyPredictor, optional
        """
        self._p_h = default_physiological_p_h
        self._ionic_strength = default_physiological_ionic_strength
        self._p_mg = default_physiological_p_mg
        self._temperature = default_physiological_temperature

        if rmse_inf is not None:
            assert rmse_inf.check(
                "[energy]/[substance]"
            ), "rmse_inf must be in units of kJ/mol (or equivalent)"

        self.ccache = ccache or create_compound_cache_from_zenodo()

        if predictor is None:
            parameters = CCModelParameters.from_zenodo(
                zenodo_doi=ZENODO_DOI_PARAMETERS
            )
            self.predictor = GibbsEnergyPredictor(
                parameters=parameters, rmse_inf=rmse_inf
            )
        else:
            self.predictor = predictor
        self.reaction_parser = make_reaction_parser()

    @property
    def p_h(self) -> Q_:
        """Get the pH."""
        return self._p_h

    @p_h.setter
    def p_h(self, value: Q_) -> None:
        """Set the pH."""
        assert value.check(""), "pH must be a unitless Quantity"
        self._p_h = value

    @property
    def p_mg(self) -> Q_:
        """Get the pMg."""
        return self._p_mg

    @p_mg.setter
    def p_mg(self, value: Q_) -> None:
        """Set the pMg."""
        assert value.check(""), "pMg must be a unitless Quantity"
        self._p_mg = value

    @property
    def ionic_strength(self) -> Q_:
        """Get the ionic strength."""
        return self._ionic_strength

    @ionic_strength.setter
    def ionic_strength(self, value: Q_) -> None:
        """Set the ionic strength."""
        assert value.check(
            "[concentration]"
        ), "ionic strength must be in units of concentration (e.g. Molar)"
        self._ionic_strength = value

    @property
    def temperature(self) -> Q_:
        """Get the temperature."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: Q_) -> None:
        """Set the temperature."""
        assert value.check(
            "[temperature]"
        ), "temperature must be in appropriate units (e.g. Kalvin)"
        self._temperature = value

    @staticmethod
    def legacy() -> "ComponentContribution":
        """Initialize a ComponentContribution object with the legacy version.

        The legacy version is intended for compatibility with older versions
        of equilibrator api (0.2.x - 0.3.1).
        Starting from 0.3.2, there is a significant change in the predictions
        caused by an improved Mg2+ concentration model.

        Returns
        -------
        A ComponentContribution object
        """
        cc = ComponentContribution.initialize_custom_version(
            zenodo_doi_params=ZENODO_DOI_PARAMETERS_LEGACY
        )
        cc.p_mg = Q_(14)  # i.e. set [Mg2+] to (essentially) zero
        return cc

    @staticmethod
    def initialize_custom_version(
        rmse_inf: Q_ = default_rmse_inf,
        zenodo_doi_cache: str = DEFAULT_ZENODO_DOI,
        zenodo_doi_params: str = ZENODO_DOI_PARAMETERS,
    ) -> "ComponentContribution":
        """Initialize a ComponentContribution object with custom quilt versions.

        Parameters
        ----------
        rmse_inf : Quantity, optional
            Set the factor by which to multiply the error
            covariance matrix for reactions outside the span of
            Component Contribution.
            (Default value: 1e-5 kJ/mol)
        zenodo_doi_cache : str, optional
            (Default value: "10.5281/zenodo.4128543")
        zenodo_doi_params : str, optional
            (Default value: "10.5281/zenodo.4013789")

        Returns
        -------
        A ComponentContribution object

        """
        ccache = create_compound_cache_from_zenodo(zenodo_doi_cache)
        parameters = CCModelParameters.from_zenodo(zenodo_doi_params)
        predictor = GibbsEnergyPredictor(
            parameters=parameters, rmse_inf=rmse_inf
        )
        return ComponentContribution(ccache=ccache, predictor=predictor)

    def get_compound(self, compound_id: str) -> Union[Compound, None]:
        """Get a Compound using the DB namespace and its accession.

        Returns
        -------
        cpd : Compound
        """
        return self.ccache.get_compound(compound_id)

    def get_compound_by_inchi(self, inchi: str) -> Union[Compound, None]:
        """Get a Compound using InChI.

        Returns
        -------
        cpd : Compound
        """
        return self.ccache.get_compound_by_inchi(inchi)

    def search_compound_by_inchi_key(self, inchi_key: str) -> List[Compound]:
        """Get a Compound using InChI.

        Returns
        -------
        cpd : Compound
        """
        return self.ccache.search_compound_by_inchi_key(inchi_key)

    def search_compound(self, query: str) -> Union[None, Compound]:
        """Try to find the compound that matches the name best.

        Parameters
        ----------
        query : str
            an (approximate) compound name

        Returns
        -------
        cpd : Compound
            the best match
        """
        hits = self.ccache.search(query)
        if not hits:
            return None
        else:
            # sort the hits by:
            # 1) Levenshtein edit-distance score (the higher the better)
            # 2) If the compound has an InChI (first the ones with InChIs)
            # 3) The compound ID in our database (the lower the better)
            hits = sorted(
                hits,
                key=lambda h: (h[1], h[0].inchi_key is not None, -h[0].id),
                reverse=True,
            )
            return hits[0][0]

    def parse_reaction_formula(self, formula: str) -> PhasedReaction:
        """Parse reaction text using exact match.

        Parameters
        ----------
        formula : str
            a string containing the reaction formula

        Returns
        -------
        rxn : Reaction
        """
        return PhasedReaction.parse_formula(self.ccache.get_compound, formula)

    def search_reaction(self, formula: str) -> PhasedReaction:
        """Search a reaction written using compound names (approximately).

        Parameters
        ----------
        formula : str
            a string containing the reaction formula

        Returns
        -------
        rxn : Reaction
        """
        results = self.reaction_parser.parseString(formula)
        substrates, arrow, products = results
        sparse = {}
        for coeff, name in substrates:
            cpd = self.search_compound(name)
            sparse[cpd] = -coeff
        for coeff, name in products:
            cpd = self.search_compound(name)
            sparse[cpd] = coeff

        return PhasedReaction(sparse, arrow=arrow)

    def balance_by_oxidation(self, reaction: PhasedReaction) -> PhasedReaction:
        """Convert an unbalanced reaction into an oxidation reaction.

        By adding H2O, O2, Pi, CO2, and NH4+ to both sides.
        """
        # We need to make sure that the number of balancing compounds is the
        # same as the number of elements we are trying to balance. Here both
        # numbers are 6, i.e. the elements are (e-, H, O, P, C, N)
        balancing_inchis = [
            "InChI=1S/p+1",  # H+
            "InChI=1S/H2O/h1H2",  # H2O
            "InChI=1S/O2/c1-2",  # O2
            "InChI=1S/H3O4P/c1-5(2,3)4/h(H3,1,2,3,4)/p-2",  # Pi
            "InChI=1S/CO2/c2-1-3",  # CO2
            "InChI=1S/H3N/h1H3/p+1",  # NH4+
        ]
        compounds = list(
            map(self.ccache.get_compound_by_inchi, balancing_inchis)
        )

        S = self.ccache.get_element_data_frame(compounds)
        balancing_atoms = S.index

        atom_bag = reaction._get_reaction_atom_bag()
        if atom_bag is None:
            logging.warning(
                "Cannot balance this reaction due to missing chemical formulas"
            )
            return reaction
        atom_vector = np.array(
            list(map(lambda a: atom_bag.get(a, 0), balancing_atoms))
        )

        other_atoms = set(atom_bag.keys()).difference(balancing_atoms)
        if other_atoms:
            raise ValueError(
                f"Cannot oxidize {reaction} only with these atoms: "
                f"{other_atoms}"
            )

        # solve the linear equation S * i = a,
        # and ignore small coefficients (< 1e-3)
        imbalance = (-np.linalg.inv(S.values) @ atom_vector).round(3)

        oxi_reaction = reaction.clone()
        for compound, coeff in zip(S.columns, imbalance.flat):
            oxi_reaction.add_stoichiometry(compound, coeff)

        return oxi_reaction

    def get_oxidation_reaction(self, compound: Compound) -> PhasedReaction:
        """Generate an oxidation Reaction for a single compound.

        Generate a Reaction object which represents the oxidation reaction
        of this compound using O2. If there are N atoms, the product must
        be NH3 (and not N2) to represent biological processes.
        Other atoms other than C, N, H, and O will raise an exception.
        """
        return self.balance_by_oxidation(PhasedReaction({compound: -1}))

    @property
    def RT(self) -> Q_:
        """Get the value of RT."""
        return R * self.temperature

    def standard_dg_formation(
        self, compound: Compound
    ) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """Get the (mu, sigma) predictions of a compound's formation energy.

        Parameters
        ----------
        compound: Compound
            a Compound object

        Returns
        -------
        mu : float
            the mean of the standard Gibbs energy of formation
        sigma : array
            the uncertainty vector

        """
        return self.predictor.get_compound_prediction(compound)

    def standard_dg(self, reaction: PhasedReaction) -> ureg.Measurement:
        """Calculate the chemical reaction energies of a reaction.

        Returns
        -------
        standard_dg : ureg.Measurement
            the dG0 in kJ/mol and standard error. To calculate the 95%
            confidence interval, use the range -1.96 to 1.96 times this value
        """
        residual_reaction, stored_dg = reaction.separate_stored_dg()

        standard_dg = self.predictor.standard_dg(residual_reaction)

        return standard_dg + stored_dg

    def standard_dg_multi(
        self,
        reactions: List[PhasedReaction],
        uncertainty_representation: str = "cov",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the chemical reaction energies of a list of reactions.

        Using the major microspecies of each of the reactants.

        Parameters
        ----------
        reactions : List[PhasedReaction]
            a list of PhasedReaction objects to estimate
        uncertainty_representation : str
            which representation to use for the uncertainties. 'cov' would
            return a full covariance matrix. 'sqrt' would return a sqaure root
            of the covariance, based on the uncertainty vectors.
            'fullrank' would return a full-rank square root of the covariance
            which is a compressed form of the 'sqrt' result.
            (Default value: 'cov')

        Returns
        -------
        standard_dg : Quantity
            the estimated standard reaction Gibbs energies based on the the
            major microspecies
        dg_uncertainty : Quantity
            the uncertainty matrix (in either 'cov', 'sqrt' or 'fullrank'
            format)
        """
        rxn_dg_pairs = map(lambda r: r.separate_stored_dg(), reactions)
        residual_reactions, stored_dg = zip(*rxn_dg_pairs)
        stored_dg = np.array(stored_dg)

        (standard_dg, dg_uncertainty,) = self.predictor.standard_dg_multi(
            residual_reactions,
            uncertainty_representation=uncertainty_representation,
        )

        standard_dg += Q_(stored_dg, "kJ/mol")
        return standard_dg, dg_uncertainty

    def standard_dg_prime(self, reaction: PhasedReaction) -> ureg.Measurement:
        """Calculate the transformed reaction energies of a reaction.

        Returns
        -------
        standard_dg : ureg.Measurement
            the dG0_prime in kJ/mol and standard error. To calculate the 95%
            confidence interval, use the range -1.96 to 1.96 times this value
        """
        residual_reaction, stored_dg_prime = reaction.separate_stored_dg_prime(
            p_h=self.p_h,
            p_mg=self.p_mg,
            ionic_strength=self.ionic_strength,
            temperature=self.temperature,
        )

        standard_dg_prime = self.predictor.standard_dg_prime(
            residual_reaction,
            p_h=self.p_h,
            p_mg=self.p_mg,
            ionic_strength=self.ionic_strength,
            temperature=self.temperature,
        )

        return standard_dg_prime + stored_dg_prime

    def dg_prime(self, reaction: PhasedReaction) -> ureg.Measurement:
        """Calculate the dG'0 of a single reaction.

        Returns
        -------
        dg : ureg.Measurement
            the dG_prime in kJ/mol and standard error. To calculate the 95%
            confidence interval, use the range -1.96 to 1.96 times this value
        """
        return (
            self.standard_dg_prime(reaction)
            + self.RT * reaction.dg_correction()
        )

    def standard_dg_prime_multi(
        self,
        reactions: List[PhasedReaction],
        uncertainty_representation: str = "cov",
    ) -> Tuple[Q_, Q_]:
        """Calculate the transformed reaction energies of a list of reactions.

        Parameters
        ----------
        reactions : List[PhasedReaction]
            a list of PhasedReaction objects to estimate
        uncertainty_representation : str
            which representation to use for the uncertainties. 'cov' would
            return a full covariance matrix. 'sqrt' would return a sqaure root
            of the covariance, based on the uncertainty vectors.
            'fullrank' would return a full-rank square root of the covariance
            which is a compressed form of the 'sqrt' result.
            (Default value: 'cov')

        Returns
        -------
        standard_dg : Quantity
            the CC estimation of the reactions' standard transformed energies
        dg_uncertainty : Quantity
            the uncertainty co-variance matrix
            (in either 'cov', 'sqrt' or 'fullrank' format)
        """
        rxn_dg_pairs = map(
            lambda r: r.separate_stored_dg_prime(
                p_h=self.p_h,
                p_mg=self.p_mg,
                ionic_strength=self.ionic_strength,
                temperature=self.temperature,
            ),
            reactions,
        )
        residual_reactions, stored_dg_primes = zip(*rxn_dg_pairs)
        stored_dg_primes = np.array(stored_dg_primes)

        (standard_dg_prime, dg_uncertainty,) = self.predictor.standard_dg_multi(
            residual_reactions,
            uncertainty_representation=uncertainty_representation,
        )

        # So far, standard_dg_prime is actually storing only the untransformed
        # standard ΔG. We must add the Legendre transform to each reaction to
        # get the standard ΔG.
        for i, r in enumerate(residual_reactions):
            try:
                standard_dg_prime[i] += r.transform(
                    p_h=self.p_h,
                    p_mg=self.p_mg,
                    ionic_strength=self.ionic_strength,
                    temperature=self.temperature,
                )
            except MissingDissociationConstantsException as e:
                warnings.warn(
                    f"Cannot calculate Legendre transform for reaction #{i}: "
                    + str(e)
                )

        standard_dg_prime += Q_(stored_dg_primes, "kJ/mol")
        return standard_dg_prime, dg_uncertainty

    def physiological_dg_prime(
        self, reaction: PhasedReaction
    ) -> ureg.Measurement:
        """Calculate the dG'm of a single reaction.

        Assume all aqueous reactants are at 1 mM, gas reactants at 1 mbar and
        the rest at their standard concentration.

        Returns
        -------
        standard_dg_primes : ndarray
            a 1D NumPy array containing the CC estimates for the reactions'
            physiological dG'
        dg_sigma : ndarray
            the second is a 2D numpy array containing the covariance matrix
            of the standard errors of the estimates. one can use the
            eigenvectors of the matrix to define a confidence high-dimensional
            space, or use dg_sigma as the covariance of a Gaussian used for
            sampling (where 'standard_dg_primes' is the mean of that Gaussian).
        """
        return (
            self.standard_dg_prime(reaction)
            + self.RT * reaction.physiological_dg_correction()
        )

    def ln_reversibility_index(
        self, reaction: PhasedReaction
    ) -> ureg.Measurement:
        """Calculate the reversibility index (ln Gamma) of a single reaction.

        Returns
        -------
        ln_RI : ureg.Measurement
            the reversibility index (in natural log scale).
        """
        physiological_dg_prime = self.physiological_dg_prime(reaction)

        abs_sum_coeff = reaction._sum_absolute_coefficients()
        if abs_sum_coeff == 0:
            return np.inf
        ln_RI = (2.0 / abs_sum_coeff) * physiological_dg_prime / self.RT
        return ln_RI

    def standard_e_prime(self, reaction: PhasedReaction) -> ureg.Measurement:
        """Calculate the E'0 of a single half-reaction.

        Returns
        -------
        standard_e_prime : ureg.Measurement
        the estimated standard electrostatic potential of reaction and
        E0_uncertainty is the standard deviation of estimation. Multiply it
        by 1.96 to get a 95% confidence interval (which is the value shown on
        eQuilibrator).
        """
        n_e = reaction.check_half_reaction_balancing()
        if n_e is None:
            raise ValueError("reaction is not chemically balanced")
        if n_e == 0:
            raise ValueError(
                "this is not a half-reaction, " "electrons are balanced"
            )
        standard_e_prime = -self.standard_dg_prime(reaction) / (n_e * FARADAY)
        return standard_e_prime.to("mV")

    def physiological_e_prime(
        self, reaction: PhasedReaction
    ) -> ureg.Measurement:
        """Calculate the E'0 of a single half-reaction.

        Returns
        -------
        physiological_e_prime : ureg.Measurement
        the estimated physiological electrostatic potential of reaction and
        E0_uncertainty is the standard deviation of estimation. Multiply it
        by 1.96 to get a 95% confidence interval (which is the value shown on
        eQuilibrator).
        """
        n_e = reaction.check_half_reaction_balancing()
        if n_e is None:
            raise ValueError("reaction is not chemically balanced")
        if n_e == 0:
            raise ValueError(
                "this is not a half-reaction, " "electrons are balanced"
            )
        return -self.physiological_dg_prime(reaction) / (n_e * FARADAY)

    def e_prime(self, reaction: PhasedReaction) -> ureg.Measurement:
        """Calculate the E'0 of a single half-reaction.

        Returns
        -------
        e_prime : ureg.Measurement
        the estimated electrostatic potential of reaction and
        E0_uncertainty is the standard deviation of estimation. Multiply it
        by 1.96 to get a 95% confidence interval (which is the value shown on
        eQuilibrator).
        """
        n_e = reaction.check_half_reaction_balancing()
        if n_e is None:
            raise ValueError("reaction is not chemically balanced")
        if n_e == 0:
            raise ValueError(
                "this is not a half-reaction, " "electrons are balanced"
            )
        return -self.dg_prime(reaction) / (n_e * FARADAY)

    def dg_analysis(self, reaction: PhasedReaction) -> List[Dict[str, object]]:
        """Get the analysis of the component contribution estimation process.

        Returns
        ------
        the analysis results as a list of dictionaries
        """
        return self.predictor.get_dg_analysis(reaction)

    def is_using_group_contribution(self, reaction: PhasedReaction) -> bool:
        """Check whether group contribution is needed to get this reactions' dG.

        Returns
        -------
        true iff group contribution is needed
        """
        return self.predictor.is_using_group_contribution(reaction)

    def multicompartmental_standard_dg_prime(
        self,
        reaction_inner: PhasedReaction,
        reaction_outer: PhasedReaction,
        delta_chi: Q_,
        p_h_outer: Q_,
        ionic_strength_outer: Q_,
        p_mg_outer: Q_ = default_physiological_p_mg,
    ) -> ureg.Measurement:
        """Calculate the transformed energies of a multi-compartmental reaction.

        Based on the equations from
        Harandsdottir et al. 2012 (https://doi.org/10.1016/j.bpj.2012.02.032)

        Parameters
        ----------
        reaction_inner : PhasedReaction
            the inner compartment half-reaction
        reaction_outer : PhasedReaction
            the outer compartment half-reaction
        delta_phi : Q_
            the difference in electro-static potential between
            the outer and inner compartments
        p_h_outer : Q_
            the pH in the outside compartment
        ionic_strength_outer : Q_
            the ionic strength outside
        p_mg_outer : Q_
            the pMg in the outside compartment

        Returns
        -------
        standard_dg_prime : ureg.Measurement
            the transport reaction Gibbs free energy change
        """

        # We want to ignore microspecies with Magnesium, as we assume they
        # never traverse the membrane.
        # Therefore, we set the value of p_mg is set to 14 (very high), so that
        # species with Mg2+ would never be the most abundant.
        p_mg = Q_(14)

        n_h_inner, n_mg_inner, z_inner = predict_protons_and_charge(
            reaction_inner,
            self.p_h,
            p_mg,
            self.ionic_strength,
            self.temperature,
        )
        n_h_outer, n_mg_outer, z_outer = predict_protons_and_charge(
            reaction_outer,
            self.p_h,
            p_mg,
            self.ionic_strength,
            self.temperature,
        )
        assert n_mg_inner == 0, "most abundant inner species have Mg2+"
        assert n_mg_outer == 0, "most abundant outer species have Mg2+"

        if (n_h_inner != -n_h_outer) or (z_inner != -z_outer):
            raise ValueError(
                "inner and outer half-reactions don't balance each other: "
                f"n_h(inner) = {n_h_inner}, n_h(outer) = {n_h_outer}, "
                f"z(inner) = {z_inner}, z(outer) = {z_outer}, "
            )

        transported_protons = n_h_outer
        transported_charge = z_outer

        (
            residual_reaction_inner,
            stored_dg_prime_inner,
        ) = reaction_inner.separate_stored_dg_prime(
            p_h=self.p_h,
            p_mg=self.p_mg,
            ionic_strength=self.ionic_strength,
            temperature=self.temperature,
        )

        (
            residual_reaction_outer,
            stored_dg_prime_outer,
        ) = reaction_outer.separate_stored_dg_prime(
            p_h=p_h_outer,
            p_mg=p_mg_outer,
            ionic_strength=ionic_strength_outer,
            temperature=self.temperature,
        )

        (standard_dg) = self.predictor.multicompartmental_standard_dg_prime(
            residual_reaction_inner,
            residual_reaction_outer,
            transported_protons,
            transported_charge,
            self.p_h,
            p_h_outer,
            self.ionic_strength,
            ionic_strength_outer,
            delta_chi,
            self.temperature,
        )

        return standard_dg + stored_dg_prime_inner + stored_dg_prime_outer

    def create_stoichiometric_matrix(
        self, reactions: Iterable[PhasedReaction]
    ) -> pd.DataFrame:
        """Build a stoichiometric matrix.

        Parameters
        ----------
        reactions : Iterable[Reaction]
            The collection of reactions to build a stoichiometric
            matrix from.

        Returns
        -------
        DataFrame
            The stoichiometric matrix as a DataFrame whose indexes are the
            compounds and its columns are the reactions (in the same order as
            the input).

        """
        return create_stoichiometric_matrix_from_reactions(
            reactions,
            self.ccache.is_proton,
            self.ccache.is_water,
            self.ccache.water,
        )

    # DISCLAIMER: this is experimental code!
    def _standard_dg_prime_multi_adjust_dof(
        self,
        reactions: List[PhasedReaction],
        uncertainty_representation: str = "cov",
    ) -> Tuple[np.array, np.array]:
        """Calculate the transformed reaction energies of a list of reactions.

        Adjusts the estimates by using the Degrees Of Freedom
        associated with the CC nullspace to minimize the norm2 of the
        standard reaction energies - using Least Squares.


        Parameters
        ----------
        reactions : List[PhasedReaction]
            a list of PhasedReaction objects to estimate
        uncertainty_representation : str
            which representation to use for the uncertainties. 'cov' would
            return a full covariance matrix. 'sqrt' would return a sqaure root
            of the covariance, based on the uncertainty vectors.
            'fullrank' would return a full-rank square root of the covariance
            which is a compressed form of the 'sqrt' result.
            (Default value: 'cov')

        Returns
        -------
        standard_dg : Quantity
            the CC estimation of the reactions' standard transformed energies
        dg_uncertainty : Quantity
            the uncertainty co-variance matrix
            (in either 'cov', 'sqrt' or 'fullrank' format)
        """
        rxn_dg_pairs = map(
            lambda r: r.separate_stored_dg_prime(
                p_h=self.p_h,
                p_mg=self.p_mg,
                ionic_strength=self.ionic_strength,
                temperature=self.temperature,
            ),
            reactions,
        )
        residual_reactions, stored_dg_primes = zip(*rxn_dg_pairs)

        (standard_dg_prime, dg_uncertainty,) = self.predictor.standard_dg_multi(
            residual_reactions,
            uncertainty_representation=uncertainty_representation,
        )

        # So far, standard_dg_prime is actually storing only the untransformed
        # standard ΔG. We must add the Legendre transform to each reaction to
        # get the standard ΔG.
        for i, r in enumerate(residual_reactions):
            try:
                standard_dg_prime[i] += r.transform(
                    p_h=self.p_h,
                    p_mg=self.p_mg,
                    ionic_strength=self.ionic_strength,
                    temperature=self.temperature,
                )
            except MissingDissociationConstantsException as e:
                warnings.warn(
                    f"Cannot calculate Legendre transform for reaction #{i}: "
                    + str(e)
                )

        standard_dg_prime += Q_(stored_dg_primes, "kJ/mol")

        # Calculate the residual matrix (X) that represents the DOFs that are
        # completely free (infinite uncertianty).
        X = self.predictor.preprocess.get_reaction_prediction_orthogonal_dof(
            residual_reactions
        )

        # Project the 'standard_dg_prime' onto the nullspace of the residual
        # matrix (using an orthogonal projection). This doesn't change the
        # values in directions that are not complete DOFs, and at the same
        # time minimizes the norm2 of the estimate vector.
        _, _, _, P_N = LINALG.invert_project(X.T)
        standard_dg_prime = P_N @ standard_dg_prime

        return standard_dg_prime, dg_uncertainty
