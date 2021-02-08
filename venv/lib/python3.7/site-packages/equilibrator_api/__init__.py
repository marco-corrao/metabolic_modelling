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

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from equilibrator_cache.thermodynamic_constants import (
    FARADAY, Q_, R, default_I, default_pH,
    default_pMg, default_T, physiological_concentration,
    standard_concentration, ureg)

default_phase = 'aqueous'

default_physiological_p_h = Q_(7.5)
default_physiological_p_mg = Q_(3.0)
default_physiological_ionic_strength = Q_(0.25, "M")
default_physiological_temperature = Q_(298.15, "K")

default_conc_lb = Q_("1e-6 M")
default_conc_ub = Q_("1e-2 M")
default_e_potential = Q_("0 V")
default_rmse_inf = Q_("1e5 kJ/mol")

from equilibrator_api.component_contribution import ComponentContribution
from equilibrator_api.phased_reaction import PhasedReaction as Reaction
