"""wrapper for component-contribution predictions."""
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
import logging
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from equilibrator_cache import FARADAY, Q_, Compound, R, Reaction, ureg
from equilibrator_cache.exceptions import MissingDissociationConstantsException
from equilibrator_cache.thermodynamic_constants import default_pMg

from . import CCModelParameters, Preprocessor
from .linalg import LINALG


logger = logging.getLogger(__name__)


class GibbsEnergyPredictor(object):
    """A class that can be used to predict dGs of reactions using CC."""

    def __init__(
        self,
        parameters: CCModelParameters = None,
        rmse_inf: Optional[Q_] = None,
    ):
        """Create a GibbsEnergyPredictor object.

        parameters : CCModelParameters
            Optional CC parameters. If not provided, the parameters are
            automatically downloaded from Zenodo.
        rmse_inf : Qunatity
            another parameter determining the uncertainty we associate
            with reactions that are not covered at all (not by RC nor GC)
            (Default value = 1e5 kJ/mol)
        """
        self.params = parameters or CCModelParameters.from_zenodo()

        if rmse_inf is not None:
            assert rmse_inf.check(
                "[energy]/[substance]"
            ), "rmse_inf must be in kJ/mol or equivalent units"
            rmse_inf = rmse_inf.m_as("kJ/mol")
            self.preprocess = Preprocessor(self.params, rmse_inf=rmse_inf)
        else:
            self.preprocess = Preprocessor(self.params)

    def get_compound_prediction(
        self, compound: Compound
    ) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
        """Get the (mu, sigma) predictions of a compound's formation energy."""
        return self.preprocess.get_compound_prediction(compound)

    def get_reaction_prediction(
        self, reaction: Reaction
    ) -> Tuple[float, np.ndarray, Dict[Compound, float]]:
        """Get the (mu, sigma) predictions of a reaction's energy.

        See the Preprocessor class for details.
        """
        return self.preprocess.get_reaction_prediction(reaction)

    def combine_uncertainties(
        self,
        sigma_fin: np.ndarray,
        sigma_inf: np.ndarray,
        sigma_res: np.ndarray,
        uncertainty_representation: str = "cov",
    ) -> Q_:
        """Calculate the transformed reaction energies of a list of reactions.

        Parameters
        ----------
        sigmas_fin : np.array
            a matrix whose columns are the square root vectors (one for each
            reaction) of the uncertainty
        sigmas_inf : np.array
            a matrix whose columns represent the infinite-uncertainty
            eigenvalues of the covariance
        sigma_res : np.array
            another matrix with the square root of the covariance corresponding
            to the 'unknown' compounds (i.e. where the uncertainty is infinite)
        uncertainty_representation : str
            which representation to use for the uncertainties. 'cov' would
            return a full covariance matrix. 'sqrt' would return a sqaure root
            of the covariance, based on the uncertainty vectors.
            'fullrank' would return a full-rank square root of the covariance
            which is a compressed form of the 'sqrt' result.
            (Default value: 'cov')

        Returns
        -------
        dg_uncertainty : Quantity
            the uncertainty co-variance matrix
            (in either 'cov', 'sqrt' or 'fullrank' format)

        """
        sigma = np.hstack(
            [
                sigma_fin,
                self.preprocess.RMSE_inf * sigma_inf,
                self.preprocess.RMSE_inf * sigma_res,
            ]
        )

        if uncertainty_representation == "cov":
            dg_uncertainty = Q_(sigma @ sigma.T, "(kJ/mol)**2")
        elif uncertainty_representation == "sqrt":
            dg_uncertainty = Q_(sigma, "kJ/mol")
        elif uncertainty_representation == "fullrank":
            dg_uncertainty = Q_(LINALG.qr_rank_deficient(sigma.T).T, "kJ/mol")
        else:
            raise ValueError(
                "uncertainty_representation must be 'cov', 'sqrt' or 'fullrank'"
            )
        return dg_uncertainty

    def standard_dgf(self, compound: Compound) -> ureg.Measurement:
        """Calculate the chemical formation energy of the major MS at pH 7.

        Parameters
        ----------
        compound: Compound :
            a compound object


        Returns
        -------
        Measurement
            a tuple of two arrays. the first is a 1D NumPy array
            containing the CC estimates for the reactions' untransformed dG0
            (i.e. using the major MS at pH 7 for each of the reactants).
            the second is a 2D numpy array containing the covariance matrix
            of the standard errors of the estimates. one can use the
            eigenvectors of the matrix to define a confidence high-dimensional
            space, or use U as the covariance of a Gaussian used for sampling
            (where dG0_cc is the mean of that Gaussian).

        """
        mu, sigma_fin, sigma_inf = self.get_compound_prediction(compound)
        if mu is None:
            return Q_(0, "kJ/mol").plus_minus(self.preprocess.RMSE_inf)
        else:
            std = np.linalg.norm(
                sigma_fin, 2
            ) + self.preprocess.RMSE_inf * np.linalg.norm(sigma_inf, 2)
            return Q_(mu, "kJ/mol").plus_minus(std)

    def standard_dg(self, reaction: Reaction) -> ureg.Measurement:
        r"""Calculate the chemical reaction energy.

        Using the major microspecies of each of the reactants.

        Parameters
        ----------
        reaction: Reaction :
            the input Reaction object


        Returns
        -------
        Measurement
        standard_dg : Measurement
            the :math:`\Delta G` in kJ/mol and standard error. to
            calculate the 95% confidence interval, multiply the error by 1.96

        """
        mu, sigma_fin, sigma_inf, residual = self.get_reaction_prediction(
            reaction
        )
        if residual:
            return Q_(0, "kJ/mol").plus_minus(self.preprocess.RMSE_inf)
        else:
            std = np.linalg.norm(
                sigma_fin, 2
            ) + self.preprocess.RMSE_inf * np.linalg.norm(sigma_inf, 2)
            return Q_(mu, "kJ/mol").plus_minus(std)

    def standard_dg_multi(
        self, reactions: List[Reaction], uncertainty_representation: str = "cov"
    ) -> Tuple[Q_, Q_]:
        r"""Calculate the chemical reaction energies for a list of reactions.

        Using the major microspecies of each of the reactants.

        Parameters
        ----------
        reactions : List[Reaction]
            a list of Reaction objects
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
            the array of Component Contribution estimates for the reactions'
            untransformed :math:`\Delta G` in kJ/mol,
            (i.e. using the major MS at pH 7 for each of the reactants).
        cov_dg : Quantity
            the covariance matrix of the standard errors of the estimates.
            one can use the eigenvectors of the matrix to define a confidence
            high-dimensional space, or use U as the covariance of a Gaussian
            used for sampling (where cov_dg is the mean of that Gaussian).

        """
        (
            mu,
            sigma_fin,
            sigma_inf,
            sigma_res,
        ) = self.preprocess.get_reaction_prediction_multi(reactions)

        dg_uncertainty = self.combine_uncertainties(
            sigma_fin, sigma_inf, sigma_res, uncertainty_representation
        )

        standard_dg = Q_(mu, "kJ/mol")

        return standard_dg, dg_uncertainty

    @ureg.check(None, None, "", "[concentration]", "[temperature]", "")
    def standard_dgf_prime(
        self,
        compound: Compound,
        p_h: Q_,
        ionic_strength: Q_,
        temperature: Q_,
        p_mg: Q_ = default_pMg,
    ) -> ureg.Measurement:
        """Calculate the biocheimcal formation energy of the compound.

        Parameters
        ----------
        compound : Compound
            a compound object
        p_h : Quantity
            the pH
        ionic_strength : Quantity
            the ionic strength
        temperature : Quantity
            temperature in Kalvin
        p_mg : Quantity, optional
            the pMg (Default value = default_pMg)


        Returns
        -------
        type
            a tuple of two arrays. the first is a 1D NumPy array
            containing the CC estimates for the reactions' untransformed dG0
            (i.e. using the major MS at pH 7 for each of the reactants).
            the second is a 2D numpy array containing the covariance matrix
            of the standard errors of the estimates. one can use the
            eigenvectors of the matrix to define a confidence high-dimensional
            space, or use U as the covariance of a Gaussian used for sampling
            (where dG0_cc is the mean of that Gaussian).

        """
        return self.standard_dgf(compound) + compound.transform(
            p_h=p_h,
            ionic_strength=ionic_strength,
            temperature=temperature,
            p_mg=p_mg,
        )

    @ureg.check(None, None, "", "[concentration]", "[temperature]", "")
    def standard_dg_prime(
        self,
        reaction: Reaction,
        p_h: Q_,
        ionic_strength: Q_,
        temperature: Q_,
        p_mg: Q_ = default_pMg,
    ) -> ureg.Measurement:
        r"""Calculate the transformed reaction energies of a reaction.

        Parameters
        ----------
        reaction : Reaction
            the input Reaction object
        p_h : Q_
            pH
        ionic_strength : Q_
            ionic strength
        temperature : Q_
            temperature
        p_mg: Q_ :
             (Default value = default_pMg)

        Returns
        -------
        standard_dg : Measurement
            the :math:`\Delta G'` in kJ/mol and standard error. to
            calculate the 95% confidence interval, multiply the error by 1.96

        """
        return self.standard_dg(reaction) + reaction.transform(
            p_h=p_h,
            ionic_strength=ionic_strength,
            temperature=temperature,
            p_mg=p_mg,
        )

    @ureg.check(None, None, "", "[concentration]", "[temperature]", "")
    def standard_dg_prime_multi(
        self,
        reactions: List[Reaction],
        p_h: Q_,
        ionic_strength: Q_,
        temperature: Q_,
        p_mg: Q_ = default_pMg,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Calculate the transformed reaction energies of a list of reactions.

        Parameters
        ----------
        reactions : List[Reaction]
            a list of Reaction objects
        p_h : Quantity
            the pH
        ionic_strength : Quantity
            the ionic strength
        temperature : Quantity
            the temperature
        p_mg: Quantity, optional
            (Default value = default_pMg)

        Returns
        -------
        standard_dg_prime : np.ndarray
            the array of Component Contribution estimates for the reactions'
            :math:`\Delta G'` in kJ/mol
        cov_dg : np.ndarray
            the covariance matrix of the standard errors of the estimates.
            one can use the eigenvectors of the matrix to define a confidence
            high-dimensional space, or use U as the covariance of a Gaussian
            used for sampling (where cov_dg is the mean of that Gaussian).

        """
        standard_dg, cov_dg = self.standard_dg_multi(reactions)

        for i, r in enumerate(reactions):
            try:
                standard_dg[i] += r.transform(
                    p_h=p_h,
                    ionic_strength=ionic_strength,
                    temperature=temperature,
                    p_mg=p_mg,
                )
            except MissingDissociationConstantsException as e:
                warnings.warn(
                    f"Cannot calculate Legendre transform for reaction #{i}: "
                    + str(e)
                )

        return standard_dg, cov_dg

    @ureg.check(
        None,
        None,
        None,
        None,
        None,
        "",
        "",
        "[concentration]",
        "[concentration]",
        "[energy]/[current]/[time]",
        "[temperature]",
        "",
        "",
    )
    def multicompartmental_standard_dg_prime(
        self,
        reaction_1: Reaction,
        reaction_2: Reaction,
        transported_protons: float,
        transported_charge: float,
        p_h_1: Q_,
        p_h_2: Q_,
        ionic_strength_1: Q_,
        ionic_strength_2: Q_,
        delta_chi: Q_,
        temperature: Q_,
        p_mg_1: Q_ = default_pMg,
        p_mg_2: Q_ = default_pMg,
    ) -> ureg.Measurement:
        r"""Calculate the transformed energies of multi-compartmental reactions.

        Based on the equations from
        Harandsdottir et al. 2012 (https://doi.org/10.1016/j.bpj.2012.02.032)

        Parameters
        ----------
        reaction_1 : Reaction
            half-reaction associated to compartment 1
        reaction_2 : Reaction
            half-reaction associated to compartment 2
        transported_protons : float
            the total number of protons
            transported through the membrane
        transported_charge : float
            the total charge
            transported through the membrane
        p_h_1 : Quantity
            the pH in compartment 1
        p_h_2 : Quantity
            the pH in compartment 2
        ionic_strength_1 : Quantity
            the ionic strength in compartment 1
        ionic_strength_2 : Quantity
            the ionic strength in compartment 2
        delta_chi : Quantity
            electrostatic potential across the membrane
        temperature : Quantity
            temperature in Kalvin
        p_mg_1 : Quantity, optional
            the pMg in compartment 1 (Default value = 10)
        p_mg_2 : Quantity, optional
            the pMg in compartment 2 (Default value = 10)

        Returns
        -------
        standard_dg : Measurement
            the :math:`\Delta G'` in kJ/mol and standard error. to
            calculate the 95% confidence interval, multiply the error by 1.96

        """
        standard_dg, cov_dg = self.standard_dg_multi([reaction_1, reaction_2])

        # the contribution of the chemical reaction to the standard ΔG' is
        # the sum of the two half reactions and the variance is the sum
        # of all the values in cov_dg. For the uncertainty we take the square
        # root of the sum.

        uncertainty = np.sqrt(sum(cov_dg.flat))
        standard_dg = (standard_dg[0] + standard_dg[1]).plus_minus(uncertainty)

        # Here we calculate the transform (since the two reactions are in
        # different pH and I, we have to do it here):
        transform_dg = Q_(0.0, "kJ/mol")
        for compound, coeff in reaction_1.items(protons=False):
            transform_dg += coeff * compound.transform(
                p_h=p_h_1,
                ionic_strength=ionic_strength_1,
                temperature=temperature,
                p_mg=p_mg_1,
            )

        for compound, coeff in reaction_2.items(protons=False):
            transform_dg += coeff * compound.transform(
                p_h=p_h_2,
                ionic_strength=ionic_strength_2,
                temperature=temperature,
                p_mg=p_mg_2,
            )

        dg_protons = (
            transported_protons
            * R
            * temperature
            * np.log(10.0)
            * (p_h_2 - p_h_1)
        )

        dg_electrostatic = FARADAY * transported_charge * delta_chi

        standard_dg += transform_dg - dg_protons - dg_electrostatic
        return standard_dg.to("kJ/mol")

    def dg_analysis(self, reaction: Reaction) -> Iterable[Dict[str, object]]:
        r"""Analyse the weight of each observation to the :math:`\Delta G`.

        Parameters
        ----------
        reaction : Reaction
            the input Reaction object


        Returns
        -------
        list
            a list of reactions that contributed to the value of the
            :math:`\Delta G` estimation, with their weights and extra
            information

        """
        return self.preprocess.dg_analysis(reaction)

    def is_using_group_contribution(self, reaction: Reaction) -> bool:
        r"""Check if group contributions is needed for the :math:`\Delta G`.

        Parameters
        ----------
        reaction : Reaction
            the input Reaction object


        Returns
        -------
        bool
            True if the reaction require group contributions.

        """
        return self.preprocess.is_using_group_contribution(reaction)
