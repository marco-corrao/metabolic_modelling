"""module for component-contribution predictions."""
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
from collections import namedtuple
from io import BytesIO
from typing import BinaryIO, Union

import numpy as np
import pandas as pd

from . import ZENODO_DOI_PARAMETERS
from .diskcache import get_cached_file


logger = logging.getLogger(__name__)

PARAMETER_NAMES = (
    "train_b train_S train_w train_G "
    "dimensions "
    "dG0_rc dG0_gc dG0_cc "
    "V_rc V_gc V_inf MSE "
    "P_R_rc P_R_gc P_N_rc P_N_gc "
    "inv_S inv_GS inv_SWS inv_GSWGS "
)


class CCModelParameters(namedtuple("CCModelParameters", PARAMETER_NAMES)):
    """Container class for all Component Contribution parameters."""

    @staticmethod
    def from_zenodo(
        zenodo_doi: str = ZENODO_DOI_PARAMETERS,
    ) -> "CCModelParameters":
        """Get the CC parameters from Zenodo.

        Parameters
        ----------
        zenodo_doi : str
            The DOI for the Zenodo entry

        Returns
        -------
        CCModelParameters
            a collection of Component Contribution parameters.

        """
        params_file = get_cached_file(
            zenodo_doi=zenodo_doi, fname="cc_params.npz"
        )
        return CCModelParameters.from_npz(params_file)

    def to_npz(self, file: Union[str, BinaryIO]) -> None:
        """Save the parameters in NumPy uncompressed .npz format."""

        # convert the CCModelParameters object into a dictionary of NumPy
        # arrays. if one of the items is a pandas DataFrames, serialize it to 3
        # separate arrays (values, index, columns)
        param_dict = dict()
        for parameter_name in self._fields:
            parameter_value = self.__getattribute__(parameter_name)
            if type(parameter_value) == pd.DataFrame:
                param_dict[f"{parameter_name}_values"] = parameter_value.values
                param_dict[f"{parameter_name}_index"] = parameter_value.index
                param_dict[
                    f"{parameter_name}_columns"
                ] = parameter_value.columns
            else:
                param_dict[parameter_name] = parameter_value
        np.savez(file, **param_dict)

    @staticmethod
    def from_npz(file: Union[str, BytesIO]) -> "CCModelParameters":
        """Load the parameters from a NumPy uncompressed .npz file."""

        npzfile = np.load(file, allow_pickle=True)
        param_dict = dict(npzfile)

        # translate the serializes DataFrames back to the original form
        for df_name in ["dimensions", "MSE", "train_G", "train_S"]:
            param_dict[df_name] = pd.DataFrame(
                data=param_dict[f"{df_name}_values"],
                index=param_dict[f"{df_name}_index"],
                columns=param_dict[f"{df_name}_columns"],
            )
            del param_dict[f"{df_name}_index"]
            del param_dict[f"{df_name}_columns"]
            del param_dict[f"{df_name}_values"]
        return CCModelParameters(**param_dict)
