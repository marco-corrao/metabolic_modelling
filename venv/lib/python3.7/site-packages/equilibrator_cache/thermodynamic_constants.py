"""A module for all thermodynamic constants and general calculations."""
# The MIT License (MIT)
#
# Copyright (c) 2018 Institute for Molecular Systems Biology, ETH Zurich.
# Copyright (c) 2019 Novo Nordisk Foundation Center for Biosustainability,
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

import os
import warnings  # Silence NEP 18 warning

import numpy as np
import pint


# Disable Pint's old fallback behavior (must come before importing Pint)
os.environ["PINT_ARRAY_PROTOCOL_FALLBACK"] = "0"

ureg = pint.UnitRegistry(system="mks")
Q_ = ureg.Quantity

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Q_([])


R = Q_(8.31e-3, "kJ / mol / K")
LOG10 = np.log(10)
FARADAY = Q_(96.485, "kC / mol")
default_T = Q_(298.15, "K")
default_I = Q_(0.25, "M")
default_pH = Q_(7.0)
default_pMg = Q_(10)
default_RT = R * default_T
default_c_mid = Q_(1e-3, "M")
default_c_range = (Q_(1e-6, "M"), Q_(1e-2, "M"))
standard_dg_formation_mg = -455.3  # ΔGf(Mg2+) at 298.15K in kJ/mol
standard_dh_formation_mg = -467.00  # ΔHf(Mg2+) at 298.15K in kJ/mol

default_T_in_K = default_T.m_as("K")
default_R_in_kJ_per_mol_per_K = R.m_as("kJ/mol/K")

standard_concentration = Q_(1.0, "M")
physiological_concentration = Q_(1.0e-3, "M")


def debye_hueckel(sqrt_ionic_strength: float, T_in_K: float) -> float:
    """Compute the ionic-strength-dependent transformation coefficient.

    For the Legendre transform to convert between chemical and biochemical
    Gibbs energies, we use the extended Debye-Hueckel theory to calculate the
    dependence on ionic strength and temperature.

    Parameters
    ----------
    sqrt_ionic_strength : float
        The square root of the ionic-strength in M
    temperature : float
        The temperature in K


    Returns
    -------
    Quantity
        The DH factor associated with the ionic strength at this
        temperature in kJ/mol

    """
    _a1 = 9.20483e-3  # kJ / mol / M^0.5 / K
    _a2 = 1.284668e-5  # kJ / mol / M^0.5 / K^2
    _a3 = 4.95199e-8  # kJ / mol / M^0.5 / K^3
    B = 1.6  # 1 / M^0.5
    alpha = _a1 * T_in_K - _a2 * T_in_K ** 2 + _a3 * T_in_K ** 3  # kJ / mol
    return alpha / (1.0 / sqrt_ionic_strength + B)  # kJ / mol


def _legendre_transform(
    pH: float,
    pMg: float,
    ionic_strength_M: float,
    T_in_K: float,
    charge: float,
    num_protons: float,
    num_magnesiums: float,
) -> float:
    RT = default_R_in_kJ_per_mol_per_K * T_in_K
    proton_term = num_protons * RT * LOG10 * pH
    _dg_mg = (T_in_K / default_T_in_K) * standard_dg_formation_mg + (
        1.0 - T_in_K / default_T_in_K
    ) * standard_dh_formation_mg
    magnesium_term = num_magnesiums * (RT * LOG10 * pMg - _dg_mg)

    if ionic_strength_M > 0:
        dh_factor = debye_hueckel(ionic_strength_M ** 0.5, T_in_K)
        is_term = dh_factor * (charge ** 2 - num_protons - 4 * num_magnesiums)
    else:
        is_term = 0.0

    return (proton_term + magnesium_term - is_term) / RT
