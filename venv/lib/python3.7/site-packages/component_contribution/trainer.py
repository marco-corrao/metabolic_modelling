"""Component Contribution Training script."""
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

import numpy as np
import pandas as pd

from . import CCModelParameters
from .linalg import LINALG
from .training_data import TrainingData


logger = logging.getLogger(__name__)


class Trainer(object):
    """A class for training the component contribution model."""

    @staticmethod
    def train(training_data: TrainingData) -> CCModelParameters:
        """Create a GibbsEnergyPredictor object.

        Parameters
        ----------
        training_data : TrainingData
            a TrainingData object containing all training data

        Returns
        -------
        CCModelParameters
            The parameters of the Component Contribution model.

        """
        G = Trainer.group_incidence_matrix(training_data)
        S = training_data.stoichiometric_matrix
        b = training_data.standard_dg
        w = training_data.weight

        # We need to convert the Compound objects in the index of S and G to
        # serializable primitives, in order to later store them in Quilt.
        # Therefore, we replace the Compound object with the Compound.id

        S.index = S.index.map(lambda c: c.id)
        G.index = G.index.map(lambda c: c.id)

        # The number of "real" groups (i.e. excluding the non-decomposable
        # compounds). This is needed to know the length of group vectors.
        number_of_groups = training_data.group_summary.shape[0]

        params = Trainer.train_from_matrices(S, G, b, w, number_of_groups)
        return params

    @staticmethod
    def group_incidence_matrix(training_data: TrainingData) -> pd.DataFrame:
        """Initialize G matrix.

        Decompose each of the compounds that has an InChI into groups,
        and save the decomposition as a row in the G matrix.
        Compounds that cannot be decomposed are added as new "groups",
        i.e. as extra columns to the matrix.

        training_data : TrainingData
            a TrainingData object containing all training data

        Returns
        -------
        pd.DataFrame
            The group incidence matrix as a pd.DataFrame.

        """
        non_decomposable_compounds = (
            training_data.non_decomposable_compounds.copy()
        )

        compound_to_gv = {}
        # decompose the compounds in the training_data and add to G
        for compound in training_data.decomposable_compounds:
            if compound.group_vector:
                compound_to_gv[compound] = compound.group_vector
            else:
                non_decomposable_compounds.add(compound)
                if compound.group_vector is None:
                    logger.error(
                        "The compound (%r) is not decomposed yet, but is "
                        "in the CC training dataset. Please run "
                        "'populate_cache 'decompose' and try again.",
                        compound,
                    )

        _group_df = training_data.group_summary

        group_data = [
            compound_to_gv.get(cpd, [0] * _group_df.shape[0])
            for cpd in training_data.compounds
        ]

        G = pd.DataFrame(
            index=training_data.compounds,
            columns=_group_df.full_name.tolist(),
            data=group_data,
            dtype=float,
        )

        for compound in non_decomposable_compounds:
            # add a new column corresponding to this non-decomposable group
            G[compound] = 0.0

            # place a single '1' for this compound group decomposition
            G.at[compound, compound] = 1.0

        return G

    @staticmethod
    def train_from_matrices(
        train_S: pd.DataFrame,
        train_G: pd.DataFrame,
        train_b: pd.Series,
        train_w: pd.Series,
        number_of_groups: int,
    ) -> CCModelParameters:
        """Estimate standard Gibbs energies of formation.

        Parameters
        ----------
        train_S : pd.DataFrame
            training data stoichiometric matrix
        train_G : pd.DataFrame
            group incidence matrix
        train_b : pd.Series
            training data reverse transformed dG'0 values
        train_w : pd.Series
            weights
        number_of_groups : int
            the number of "real" groups

        Returns
        -------
        CCModelParameters
            Collection of Component Contribution parameters trained using
            the input data.

        """
        assert (train_G.index == train_S.index).all()
        assert (train_b.index == train_S.columns).all()
        assert (train_w.index == train_S.columns).all()

        S = train_S.values
        G = train_G.values
        b = np.array([dg.m_as("kJ/mol") for dg in train_b.tolist()], ndmin=1)
        w = np.array(train_w.tolist(), ndmin=1)

        # Apply weighing
        W = np.diag(w.ravel())
        GS = G.T @ S

        # Linear regression for the reactant layer (aka RC)
        inv_S, r_rc, P_R_rc, P_N_rc = LINALG.invert_project(S @ W)

        # Linear regression for the group layer (aka GC)
        inv_GS, r_gc, P_R_gc, P_N_gc = LINALG.invert_project(GS @ W)

        # calculate the group contributions
        dG0_gc = inv_GS.T @ W @ b

        # Calculate the contributions in the stoichiometric space
        dG0_rc = inv_S.T @ W @ b
        dG0_cc = P_R_rc @ dG0_rc + P_N_rc @ G @ dG0_gc

        # Calculate the residual error (unweighted squared error divided
        # by N - rank)
        e_rc = S.T @ dG0_rc - b
        MSE_rc = (e_rc.T @ W @ e_rc) / (S.shape[1] - r_rc)

        e_gc = GS.T @ dG0_gc - b
        MSE_gc = (e_gc.T @ W @ e_gc) / (S.shape[1] - r_gc)

        # Calculate the MSE of GC residuals for all reactions in ker(G).
        # This will help later to give an estimate of the uncertainty for such
        # reactions, which otherwise would have a 0 uncertainty in the GC
        # method.
        kerG_inds = list(np.where(np.all(GS == 0, 0))[0].flat)

        e_kerG = e_gc[kerG_inds]
        MSE_kerG = (e_kerG.T @ e_kerG) / len(kerG_inds)

        # Calculate the uncertainty covariance matrices
        inv_SWS, _, _, _ = LINALG.invert_project(S @ W @ S.T)
        inv_GSWGS, _, _, _ = LINALG.invert_project(GS @ W @ GS.T)

        V_rc = P_R_rc @ inv_SWS @ P_R_rc
        V_gc = P_N_rc @ G @ inv_GSWGS @ G.T @ P_N_rc
        V_inf = P_N_rc @ G @ P_N_gc @ G.T @ P_N_rc

        MSE = pd.DataFrame(
            index=["rc", "gc", "kerG"],
            columns=["MSE"],
            data=[MSE_rc, MSE_gc, MSE_kerG],
        )

        dimensions = pd.DataFrame(
            index=["Nc", "Ng", "Ng_full"],
            columns=["number"],
            data=[train_G.shape[0], number_of_groups, train_G.shape[1]],
        )

        # Put all the calculated data in 'params' for the sake of debugging
        return CCModelParameters(
            train_b=b,
            train_S=train_S,
            train_w=w,
            train_G=train_G,
            dimensions=dimensions,
            dG0_rc=dG0_rc,
            dG0_gc=dG0_gc,
            dG0_cc=dG0_cc,
            V_rc=V_rc,
            V_gc=V_gc,
            V_inf=V_inf,
            MSE=MSE,
            P_R_rc=P_R_rc,
            P_R_gc=P_R_gc,
            P_N_rc=P_N_rc,
            P_N_gc=P_N_gc,
            inv_S=inv_S,
            inv_GS=inv_GS,
            inv_SWS=inv_SWS,
            inv_GSWGS=inv_GSWGS,
        )
