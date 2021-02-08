"""Define lower and upper bounds on compounds."""
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


from typing import Dict, Iterable, TextIO, Tuple

import numpy as np
import pandas as pd
import pkg_resources
from equilibrator_cache import Compound

from .. import (
    Q_,
    ComponentContribution,
    default_conc_lb,
    default_conc_ub,
    standard_concentration,
    ureg,
)


class BaseBounds(object):
    """A base class for declaring bounds on things."""

    def copy(self):
        """Return a (deep) copy of self."""
        raise NotImplementedError

    def get_lower_bound(self, compound: Compound):
        """Get the lower bound for this key.

        :param key: a compound
        """
        raise NotImplementedError

    def get_upper_bound(self, compound: Compound):
        """Get the upper bound for this key.

        :param key: a compound
        """
        raise NotImplementedError

    def get_lower_bounds(self, compounds: Iterable[Compound]) -> Iterable[Q_]:
        """Get the bounds for a set of keys in order.

        :param compounds: an iterable of Compounds
        :return:  an iterable of the lower bounds
        """
        return map(self.get_lower_bound, compounds)

    def get_upper_bounds(self, compounds: Iterable[Compound]) -> Iterable[Q_]:
        """Get the bounds for a set of keys in order.

        :param compounds: an iterable of Compounds
        :return:  an iterable of the upper bounds
        """
        return map(self.get_upper_bound, compounds)

    def get_bound_tuple(self, compound: Compound) -> Tuple[Q_, Q_]:
        """Get both upper and lower bounds for this key.

        :param compound: a Compound object
        :return: a 2-tuple (lower bound, upper bound)
        """
        return self.get_lower_bound(compound), self.get_upper_bound(compound)

    def get_bounds(
        self, compounds: Iterable[Compound]
    ) -> Tuple[Iterable[Q_], Iterable[Q_]]:
        """Get the bounds for a set of compounds.

        :param compounds: an iterable of Compounds
        :return: a 2-tuple (lower bounds, upper bounds)
        """
        bounds = map(self.get_bound_tuple, compounds)
        lbs, ubs = zip(*bounds)
        return lbs, ubs

    @staticmethod
    @ureg.check("[concentration]")
    def conc2ln_conc(b: Q_) -> float:
        """Convert a concentration to log-concentration.

        :param b: a concentration
        :return: the log concentration
        """
        return np.log((b / standard_concentration).m_as(""))

    def get_ln_bounds(
        self, compounds: Iterable[Compound]
    ) -> Tuple[Iterable[float], Iterable[float]]:
        """Get the log-bounds for a set of compounds.

        :param compounds: an iterable of Compounds
        :return: a 2-tuple (log lower bounds, log upper bounds)
        """
        lbs, ubs = self.get_bounds(compounds)

        return map(self.conc2ln_conc, lbs), map(self.conc2ln_conc, ubs)

    def get_ln_lower_bounds(
        self, compounds: Iterable[Compound]
    ) -> Iterable[float]:
        """Get the log lower bounds for a set of compounds.

        :param compounds: an iterable of Compounds
        :return: an iterable of log lower bounds
        """
        lbs = self.get_lower_bounds(compounds)
        return map(self.conc2ln_conc, lbs)

    def get_ln_upper_bounds(
        self, compounds: Iterable[Compound]
    ) -> Iterable[float]:
        """Get the log upper bounds for a set of compounds.

        :param compounds: an iterable of Compounds
        :return: an iterable of log upper bounds
        """
        ubs = self.get_upper_bounds(compounds)
        return map(self.conc2ln_conc, ubs)

    @ureg.check(None, None, "[concentration]", "[concentration]")
    def set_bounds(self, compound: Compound, lb: Q_, ub: Q_) -> None:
        """Set bounds for a specific key.

        :param key: a string representation of a KEGG compound ID,
        i.e. C00001 for water
        :param lb: the lower bound value
        :param ub: the upper bound value
        """
        assert lb <= ub
        self.lower_bounds[compound] = lb
        self.upper_bounds[compound] = ub


class Bounds(BaseBounds):
    """Contains upper and lower bounds for various keys.

    Allows for defaults.

    """

    @ureg.check(None, None, None, "[concentration]", "[concentration]")
    def __init__(
        self,
        lower_bounds: Dict[Compound, Q_] = None,
        upper_bounds: Dict[Compound, Q_] = None,
        default_lb: Q_ = default_conc_lb,
        default_ub: Q_ = default_conc_ub,
    ) -> None:
        """Initialize the bounds object.

        :param lower_bounds: a dictionary mapping keys to lower bounds
        :param upper_bounds: a dictionary mapping keys to upper bounds
        :param default_lb: default lower bound to if no specific one is
        provided
        :param default_lb: default upper bound to if no specific one is
        provided
        """
        self.lower_bounds = lower_bounds or dict()
        self.upper_bounds = upper_bounds or dict()
        for b in self.lower_bounds.values():
            assert b.check("[concentration]")
        for b in self.upper_bounds.values():
            assert b.check("[concentration]")

        self.default_lb = default_lb
        self.default_ub = default_ub

    @classmethod
    @ureg.check(None, None, None, "[concentration]", "[concentration]")
    def from_csv(
        cls,
        f: TextIO,
        comp_contrib: ComponentContribution,
        default_lb: Q_ = default_conc_lb,
        default_ub: Q_ = default_conc_ub,
    ) -> "Bounds":
        """Read Bounds from a CSV file.

        Parameters
        ----------
        f : TextIO
            an open .csv file stream
        comp_contrib : ComponentContribution
            used for parsing compound accessions
        default_lb : Q_
            the default lower bound
        default_ub : Q_
            the default upper bound
        """
        lbs: Dict[Compound, Q_] = dict()
        ubs: Dict[Compound, Q_] = dict()
        bounds_df = pd.read_csv(f)

        for row in bounds_df.itertuples():
            compound = comp_contrib.get_compound(row.compound_id)
            if compound is None:
                raise ValueError(
                    f"Cannot find this compound accession: "
                    f"{row.compound_id}"
                )
            lbs[compound] = row.lb * ureg.molar
            ubs[compound] = row.ub * ureg.molar

        bounds = Bounds(lbs, ubs, default_lb, default_ub)
        bounds.check_bounds()
        return bounds

    def to_data_frame(self) -> pd.DataFrame:
        """Convert the list of bounds to a Pandas DataFrame."""
        data = [
            (c, self.get_lower_bound(c), self.get_upper_bound(c))
            for c in self.lower_bounds.keys()
        ]
        return pd.DataFrame(data=data, columns=["compound", "lb", "ub"])

    def check_bounds(self) -> None:
        """Assert the bounds are valid (i.e. that lb <= ub)."""
        assert self.default_lb <= self.default_ub, (
            f"default lower bound ({self.default_lb}) is higher than the "
            f"default upper bound ({self.default_ub})"
        )

        for compound in self.upper_bounds:
            lb = self.get_lower_bound(compound)
            ub = self.get_upper_bound(compound)
            assert lb <= ub, (
                f"lower bound ({self.default_lb}) for {compound} is higher "
                f"than the upper bound ({self.default_ub})"
            )

    def copy(self) -> "Bounds":
        """Return a deep copy of self."""
        return Bounds(
            self.lower_bounds.copy(),
            self.upper_bounds.copy(),
            self.default_lb,
            self.default_ub,
        )

    def get_lower_bound(self, compound: Compound) -> Q_:
        """Get the lower bound for this compound."""
        return self.lower_bounds.get(compound, self.default_lb)

    def get_upper_bound(self, compound: Compound) -> Q_:
        """Get the upper bound for this compound."""
        return self.upper_bounds.get(compound, self.default_ub)

    # the default is generated only at the first call of GetDefaultBounds()
    # the reason to do this, is because looking up all the co-factor compounds
    # in the cache takes about 20 seconds
    DEFAULT_BOUNDS = None

    @staticmethod
    def get_default_bounds(comp_contrib: ComponentContribution) -> "Bounds":
        """Return the default lower and upper bounds for a pre-determined list.

        Parameters
        ----------
        comp_contrib : ComponentContribution

        Returns
        -------
        a Bounds object with the default values
        """
        if Bounds.DEFAULT_BOUNDS is None:
            COFACTORS_FNAME = pkg_resources.resource_filename(
                "equilibrator_api", "data/cofactors.csv"
            )
            with open(COFACTORS_FNAME, "r") as fp:
                Bounds.DEFAULT_BOUNDS = Bounds.from_csv(fp, comp_contrib)
        return Bounds.DEFAULT_BOUNDS
