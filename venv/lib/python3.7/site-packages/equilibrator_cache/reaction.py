"""A biochemical reaction module."""
# The MIT License (MIT)
#
# Copyright (c) 2018 Institute for Molecular Systems Biology, ETH Zurich.
# Copyright (c) 2018 Novo Nordisk Foundation Center for Biosustainability,
# Technical University of Denmark.
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
import re
import sys
import warnings
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import PROTON_INCHI, WATER_INCHI
from .exceptions import ParseException
from .models import Compound
from .thermodynamic_constants import Q_, default_pMg, ureg


__all__ = ("POSSIBLE_REACTION_ARROWS", "Reaction")


POSSIBLE_REACTION_ARROWS = (
    # Three-character arrows.
    "<=>",
    "<->",
    "-->",
    "<--",
    # Two-character arrows.
    "=>",
    "<=",
    "->",
    "<-",
    # Single character unicode arrows.
    "=",
    "⇌",
    "⇀",
    "⇋",
    "↽",
)


class Reaction:
    """A class for handling biochemical reactions."""

    def __init__(
        self,
        sparse: Dict[Compound, float],
        arrow: Optional[str] = None,
        rid: Optional[str] = None,
    ):
        """Create a new Reaction object."""
        self.sparse = {k: v for k, v in sparse.items() if v != 0}
        self.arrow = arrow or POSSIBLE_REACTION_ARROWS[0]
        self.rid = rid

    def __len__(self) -> int:
        """Count the number of reactants."""
        return len(self.sparse)

    def clone(self) -> "Reaction":
        """Create a copy of this reaction."""
        return Reaction(self.sparse, self.arrow, self.rid)

    def keys(self, protons=True, water=True) -> Iterable[Compound]:
        """Iterate over the reactants.

        Parameters
        ----------
        protons : bool
             Include H+ as one of the keys (Default value = True)
        water : bool
             Include H2O as one of the keys (Default value = True)

        Returns
        -------
        Iterable[Compound]
            An iterator over all reactants.

        """
        for c in self.sparse.keys():
            if not water and c.inchi == WATER_INCHI:
                continue
            if not protons and c.inchi == PROTON_INCHI:
                continue
            yield c

    def items(
        self, protons=True, water=True
    ) -> Iterable[Tuple[Compound, float]]:
        """Iterate over the reactants and their stoichiometric coefficients.

        Parameters
        ----------
        protons : bool
             Include H+ as one of the keys (Default value = True)
        water : bool
             Include H2O as one of the keys (Default value = True)

        Returns
        -------
        Iterable[Tuple[Compound, int]]
            An iterator of all reactants where the values are tuples of
            Compound and stoichiometric coefficient

        """
        for c, i in self.sparse.items():
            if not water and c.inchi == WATER_INCHI:
                continue
            if not protons and c.inchi == PROTON_INCHI:
                continue
            yield c, i

    def get_coeff(self, compound: Compound) -> float:
        """Get the stoichiometric coefficient of a reactant.

        Parameters
        ----------
        compound : Compound
            A reactant Compound object

        Returns
        -------
        float
            The stoichiometric coefficient.


        """
        return self.sparse.get(compound, 0.0)

    def reverse(self) -> "Reaction":
        """Reverse the direction of the reaction.

        The reverse reaction is calculated by negating all stoichiometric
        coefficients.


        Returns
        -------
        Reaction
            The reverse reaction.

        """

        sparse = dict((k, -v) for (k, v) in self.sparse.items())
        return Reaction(sparse, self.arrow, self.rid)

    def __eq__(self, other: "Reaction") -> bool:
        """Check if two reactions are equal."""
        cpds = set(self.keys(protons=False)).union(other.keys(protons=False))
        for c in cpds:
            if self.get_coeff(c) != other.get_coeff(c):
                return False
        return True

    @staticmethod
    def parse_formula_side(
        s: str, str_to_compound: Callable[[str], Compound]
    ) -> Dict[Compound, float]:
        """Parse one side of the reaction formula.

        For example, '2 kegg:C00001 + kegg:C00002 + 3 kegg:C00003'.
        Ignores stoichiometry.

        Parameters
        ----------
        s: str :

        str_to_compound: Callable[[str] :

        Compound] :


        Returns
        -------
        dict
            The set of CIDs.

        """
        if s.strip() == "null":
            return {}

        compound_bag = {}
        for member in re.split(r"\s+\+\s+", s):
            tokens = member.split(None, 1)
            if len(tokens) == 0:
                continue
            if len(tokens) == 1:
                amount = 1.0
                compound = str_to_compound(member)
            else:
                try:
                    amount = float(tokens[0])
                except ValueError:
                    raise ParseException(f"Non-specific reaction: {s}")
                compound = str_to_compound(tokens[1])

            if compound is None:
                raise ParseException(
                    f"{member} was not found in the compound cache"
                )

            compound_bag[compound] = compound_bag.get(compound, 0.0) + amount

        return compound_bag

    @classmethod
    def parse_formula(
        cls,
        str_to_compound: Callable[[str], Compound],
        formula: str,
        rid: Optional[str] = None,
    ) -> "Reaction":
        """Parse a two-sided reaction formula.

        Examples
        --------
        >>> from equilibrator_cache import Reaction
        >>> Reaction.parse_formula(parse_compound, '2 C00001 = C00002 + C00003')


        Parameters
        ----------
        str_to_compound : Callable[[str], Compound]
            A function that returns a compound instance
            from a string, for example, from a database identifier.
        formula : str
            A string representation of the reaction.
        rid : str, optional
            An identifier for the resulting reaction object.
            (Default value = None)

        Returns
        -------
        Reaction
            An object representation of the substrates, products and the
            reaction direction.

        """
        tokens = []
        arrow = None
        for arrow in POSSIBLE_REACTION_ARROWS:
            if formula.find(arrow) != -1:
                tokens = formula.split(arrow, 2)
                break

        if len(tokens) < 2:
            raise ParseException(
                f"Reaction does not contain an allowed arrow sign ({arrow}): "
                f" {formula}"
            )

        left = tokens[0].strip()
        right = tokens[1].strip()

        sparse_reaction = {}
        left_dict = Reaction.parse_formula_side(left, str_to_compound)
        right_dict = Reaction.parse_formula_side(right, str_to_compound)
        for cid, count in left_dict.items():
            sparse_reaction[cid] = sparse_reaction.get(cid, 0) - count

        for cid, count in right_dict.items():
            sparse_reaction[cid] = sparse_reaction.get(cid, 0) + count

        # remove compounds that are balanced out in the reaction,
        # i.e. their coefficient is 0
        sparse_reaction = dict(
            filter(lambda x: x[1] != 0, sparse_reaction.items())
        )
        return cls(sparse_reaction, arrow=arrow, rid=rid)

    @staticmethod
    def write_compound_and_coeff(compound: Compound, coeff: float) -> str:
        """Write the compound preceded by the stoichiometric coefficient.

        Parameters
        ----------
        compound : Compound
            A Compound object

        coeff : float
            The stoichiometric coefficient


        Returns
        -------
        str
            A string representation of the compound preceded by the
            stoichiometric coefficient.

        """
        if np.abs(coeff - 1) < sys.float_info.epsilon:
            return str(compound)
        else:
            return "%g %s" % (coeff, str(compound))

    def __str__(self) -> str:
        """Return a string representation of the entire reaction."""
        left = []
        right = []
        for compound, coeff in sorted(self.sparse.items()):
            if coeff < 0:
                left.append(Reaction.write_compound_and_coeff(compound, -coeff))
            elif coeff > 0:
                right.append(Reaction.write_compound_and_coeff(compound, coeff))
        return f"{' + '.join(left)} {self.arrow} {' + '.join(right)}"

    def get_element_data_frame(self) -> pd.DataFrame:
        """Tabulate the elemental composition of all reactants.

        Returns
        -------
        DataFrame
            A data frame where the columns are the compounds and the
            indexes are atomic elements.
        """
        atom_bags = {
            compound: compound.atom_bag or {} for compound in self.keys()
        }

        # create the elemental matrix, where each row is a compound and each
        # column is an element (or e-)
        return pd.DataFrame.from_dict(
            atom_bags, orient="columns", dtype=int
        ).fillna(0)

    def _get_reaction_atom_bag(
        self, minimal_stoichiometry: Optional[float] = None
    ) -> Dict[str, int]:
        """Use for checking if all elements are conserved.

        Parameters
        ----------
        minimal_stoichiometry : float, optional
            if not None, sets the minimal value
            for a non-zero stoichiometry
            (Default value = None)


        Returns
        -------
        dict
            An atom_bag of the differences between the sides of the
            reaction. E.g. if there is one extra C on the left-hand
            side, the result will be {'C': -1}.

        """
        element_df = self.get_element_data_frame()

        if np.any((element_df == 0).all(axis=0)):
            warning_str = (
                "cannot generate the reaction atom bag because "
                + "compounds have unspecific formulas: "
                + "%s" % str(self)
            )
            warnings.warn(warning_str)
            return None

        stoichiometry = np.array(list(element_df.columns.map(self.get_coeff)))
        unbalanced = element_df @ stoichiometry
        unbalanced = unbalanced[unbalanced != 0]
        atom_bag = unbalanced.to_dict()

        if minimal_stoichiometry is not None:
            # ignore the differences if they are very close to 0
            atom_bag = {
                k: v
                for k, v in atom_bag.items()
                if abs(v) > minimal_stoichiometry
            }

        if len(atom_bag) > 0:
            logging.debug("unbalanced reaction: %s" % str(self))
            for elem, imbalance in atom_bag.items():
                logging.debug(
                    "there are %d more %s atoms on the "
                    "right-hand side" % (imbalance, elem)
                )

        return atom_bag

    def add_stoichiometry(self, cpd: Compound, coeff: float) -> None:
        """Add to an existing stoichiometric coefficient.

        If the compound is not part of the reaction, add it as a new reactant.

        Parameters
        ----------
        cpd : Compound
            the reactant
        coeff : float
            how much to add to its stoichiometric coefficient


        """
        if cpd in self.sparse:
            if self.sparse[cpd] == -coeff:
                self.sparse.pop(cpd)
            else:
                self.sparse[cpd] += coeff
        else:
            self.sparse[cpd] = coeff

    def is_balanced(self, ignore_atoms: Tuple[str] = ("H",)) -> bool:
        """Check if a reaction is balanced.

        Parameters
        ----------
        ignore_atoms : tuple, optional
            a tuple with all the elements to ignore
            (Default value = ("H",) )

        Returns
        -------
        bool
            True if the reaction is balanced.

        """
        reaction_atom_bag = self._get_reaction_atom_bag()

        if reaction_atom_bag is None:
            # this means some compound formulas are missing
            return False

        for atom in ignore_atoms:
            if atom in reaction_atom_bag:
                reaction_atom_bag.pop(atom)

        return len(reaction_atom_bag) == 0

    def balance_with_compound(
        self, compound: Compound, ignore_atoms: Tuple[str] = ("H",)
    ) -> Union["Reaction", None]:
        """Try to balance the reaction using a specified compound.

        Parameters
        ----------
        compound : Compound
            the compound to use for balancing
        ignore_atoms : Tuple[str]
            a tuple with all the elements to ignore
            (Default value = ("H",) )

        Returns
        -------
        Union[Reaction, None]
            A balanced reaction. If the original reaction is already
            balanced, returns it back. Otherwise, return a new balanced
            reaction, or None if the reaction cannot be balanced.

        """

        def get_pivot_atom(_compound: Compound) -> Tuple[str, float]:
            """Select one atom from the given compound and try to balance it.

            Skips the elements in `ignore_atoms`

            Parameters
            ----------
            _compound : Compound
                the compound to use for balancing


            Returns
            -------
            atom : str
                the symbol of the element used for balancing
            count : float
                the number of atoms of this type in `compound`
            """

            if _compound.atom_bag is None:
                raise Exception(
                    f"Cannot balance using this compound, it has no "
                    f"formula: {_compound.formula}"
                )

            for atom, count in _compound.atom_bag.items():
                if ignore_atoms is None or atom not in ignore_atoms:
                    return atom, count
            raise Exception(
                f"Cannot balance using this compound, it has no "
                f"relevant atoms: {_compound.formula}"
            )

        if self.is_balanced(ignore_atoms):
            return self

        reaction_atom_bag = self._get_reaction_atom_bag()

        if reaction_atom_bag is None:
            # this means some compound formulas are missing
            return None

        pivot_atom, pivot_count = get_pivot_atom(compound)

        new_reaction = self.clone()
        if pivot_atom in reaction_atom_bag:
            new_reaction.add_stoichiometry(
                compound, -reaction_atom_bag[pivot_atom] / float(pivot_count)
            )

        if new_reaction.is_balanced(ignore_atoms):
            return new_reaction

    def is_empty(self) -> bool:
        """Check if the reaction is empty.

        Returns
        -------
        bool
            whether the reaction is empty (i.e. has no reactants).

        """
        return len(self.sparse) == 0

    def dense(self, cids: List[Compound]) -> np.ndarray:
        """Return a dense representation of the reaction.

        Parameters
        ----------
        cids : List[Compound]
            The order of compounds for determining the indices in the result


        Returns
        -------
        np.ndarray
            An array with the stoichiometric coefficients in "dense" format.

        """
        s = np.zeros((len(cids), 1))
        for cid, coeff in self.items():
            s[cids.index(cid), 0] = coeff
        return s

    def can_be_transformed(self) -> bool:
        """Check if this reaction can be transformed.

        In other words, all the reactants can be transformed (i.e. have
        microspecies).

        Returns
        -------
        True if the reaction can be transformed

        """
        for compound, _ in self.items():
            if not compound.can_be_transformed():
                return False
        return True

    @ureg.check(None, "", "[concentration]", "[temperature]", "")
    def transform(
        self,
        p_h: Q_,
        ionic_strength: Q_,
        temperature: Q_,
        p_mg: Q_ = default_pMg,
    ) -> Q_:
        r"""Calculate the Legendre transform for a reaction.

        Apply the Legendre transform on each of the compounds and sum up their
        contributions (based on the stoichiometry).

        Parameters
        ----------
        p_h : Quantity
            Set the -log10 of the proton concentration [H+].
        ionic_strength : Quantity
            Set the ionic strength.
        temperature : Quantity
            Set the temperature.
        p_mg : Quantity
            Set the -log10 of the magnesium ion concentration [Mg++].
            (Default value = default_pMg)

        Returns
        -------
        Quantity
            The transformed relative :math:`\Delta G`.

        """
        ddg = Q_(0.0, "kJ/mol")
        for compound, coeff in self.items(protons=False):  # ignore protons
            ddg += coeff * compound.transform(
                p_h=p_h,
                ionic_strength=ionic_strength,
                temperature=temperature,
                p_mg=p_mg,
            )
        return ddg

    def _sum_coefficients(self) -> float:
        r"""Calculate the sum of all coefficients (excluding water).

        This is useful for shifting the :math:`\Delta G0`to another set
        of standard concentrations (e.g. 1 mM).


        Returns
        -------
        float
            The sum of all coefficients.
        """

        return sum(c for _, c in self.items(protons=False, water=False))

    def _sum_absolute_coefficients(self) -> float:
        """Calculate sum of coefficients (excluding water) in absolute value.

        This is useful for calculating the reversibility index.

        Returns
        -------
        float
            The sum of coefficients in absolute value.

        """

        return sum(abs(c) for _, c in self.items(protons=False, water=False))

    def check_half_reaction_balancing(self) -> Union[int, None]:
        """Check if this is a balanced half-reaction (redox).

        :return:

        Returns
        -------
        int or None
            The number of electrons that are 'missing' in the
            half-reaction or None if the reaction is not atomwise-balanced.
        """

        atom_bag = self._get_reaction_atom_bag()
        if atom_bag is None:
            return None

        # we ignore proton balancing
        atom_bag.pop("H", 0)

        n_e = atom_bag.pop("e-", 0)
        if len(atom_bag) > 0:
            return None
        else:
            return n_e

    @staticmethod
    def _hashable_reactants(
        sparse: Dict[Compound, float]
    ) -> Tuple[Tuple[int, float]]:
        """Return a unique list of number pairs representing reaction.

        The list fully identifies the biochemical reaction.
        If it is equal to another reaction's string, then they have identical
        stoichiometry.

        Parameters
        ----------
        sparse : Dict[Compound, float]
            a dictionary whose keys are compounds and values are
            stoichiometric coefficients


        Returns
        -------
        Tuple[Tuple[int, float]]
            a tuple containing all pairs of (compound.id, coefficient)

        """
        if len(sparse) == 0:
            return tuple()

        # sort according to compound ID and normalize the stoichiometric
        # coefficients such that the coeff of the reactant with the lowest
        # ID will be 1
        sorted_compound_list = sorted(sparse.keys(), key=lambda c: c.id)
        sorted_cpd_coeff_list = zip(
            sorted_compound_list, map(sparse.get, sorted_compound_list)
        )

        # filter out compounds with zero coefficients
        sorted_cpd_coeff_list = list(
            filter(lambda x: x[1] != 0, sorted_cpd_coeff_list)
        )
        if len(sorted_cpd_coeff_list) == 0:
            warnings.warn("All stoichiometric coefficients are 0")
            return tuple()
        norm_factor = 1.0 / sorted_cpd_coeff_list[0][1]
        r_list = [
            (cpd.id, norm_factor * coeff)
            for cpd, coeff in sorted_cpd_coeff_list
        ]
        return tuple(r_list)

    def __hash__(self) -> int:
        """Return a hash of the Reaction.

        This hash is useful for finding reactions with the exact same
        stoichiometry. We create a unique formula string based on the
        Compound IDs and coefficients.

        Returns
        -------
        int
            A hash of the Reaction.

        """
        reactants = Reaction._hashable_reactants(self.sparse)
        return hash(reactants)


def create_stoichiometric_matrix_from_reactions(
    reactions: Iterable[Reaction],
    is_proton: Callable[[Compound], bool],
    is_water: Callable[[Compound], bool],
    water: Compound,
) -> pd.DataFrame:
    """Build a stoichiometric matrix.

    Parameters
    ----------
    reactions : Iterable[Reaction]
        The collection of reactions to build a stoichiometric
        matrix from.
    is_proton : Callable[[Compound], bool]
        A function that identifies if a Compound is a proton.
    is_water : Callable[[Compound], bool]
        A function that identifies if a Compound is water.
    water : Compound
        The water compound (is added to the stoichiometric matrix
        if water is not one of the compounds in the input reaction list.


    Returns
    -------
    DataFrame
        The stoichiometric matrix as a DataFrame whose indexes are the
        compounds and its columns are the reactions (in the same order as
        the input).

    """
    # Now get rid of the protons, since we are applying Alberty's
    # framework where their potential is set to 0, and the pH is held
    # as a controlled parameter.
    compounds = {c for r in reactions for c in r.sparse if not is_proton(c)}
    if not any(map(is_water, compounds)):
        compounds.add(water)
    compounds = sorted(compounds)

    sparse_reactions = []
    for rxn in reactions:
        sparse = pd.Series(index=compounds, dtype=float)
        for met, coeff in rxn.items(protons=False):
            sparse[met] = coeff
        sparse_reactions.append(sparse)
    stoichiometry = pd.concat(sparse_reactions, axis=1).fillna(0.0)
    return stoichiometry
