"""inherit from equilibrator_cache.reaction.Reaction an add phases."""
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


import hashlib
from typing import Dict, List, Tuple

from equilibrator_cache import Compound, Reaction

from . import Q_, ureg
from .phased_compound import PhasedCompound


class PhasedReaction(Reaction):
    """A daughter class of Reaction that adds phases and abundances."""

    REACTION_COUNTER = 0

    def __init__(
        self,
        sparse: Dict[Compound, float],
        arrow: str = "<=>",
        rid: str = None,
        sparse_with_phases: Dict[PhasedCompound, float] = None,
    ) -> None:
        """Create a PhasedReaction object.

        :param sparse: a dictionary of Compounds to stoichiometric coefficients
        :param arrow: a string representing the arrow in the chemical formula
        :param rid: a string of the reaction ID
        """
        super(PhasedReaction, self).__init__(sparse, arrow, rid)

        if sparse_with_phases is not None:
            self.sparse_with_phases = sparse_with_phases
        else:
            self.sparse_with_phases = {
                PhasedCompound(cpd): coeff for cpd, coeff in sparse.items()
            }

        if self.rid is None:
            self.rid = "COCO:R%05d" % PhasedReaction.REACTION_COUNTER
            PhasedReaction.REACTION_COUNTER += 1

    def clone(self) -> "PhasedReaction":
        """Clone this reaction object."""
        return PhasedReaction(
            self.sparse.copy(),
            self.arrow,
            self.rid,
            self.sparse_with_phases.copy(),
        )

    def reverse(self) -> "PhasedReaction":
        """Return a PhasedReaction with the reverse reaction."""
        sparse = {cpd: -coeff for (cpd, coeff) in self.sparse.items()}

        sparse_with_phases = {
            cpd: -coeff for (cpd, coeff) in self.sparse_with_phases.items()
        }

        return PhasedReaction(sparse, self.arrow, self.rid, sparse_with_phases)

    @staticmethod
    def _hashable_reactants(
        sparse_with_phases: Dict[PhasedCompound, float],
        reversible: bool = False,
    ) -> Tuple[Tuple[int, str, float]]:
        """Return a unique nested tuple representing this reaction.

        Parameters
        ----------
        sparse_with_phases: dict
            a dictionary containing the compounds and stoichiometry
        reversible : bool
            a flag indicating whether the directionality of the reaction
            matters or not. if `True`, the same value will be returned for
            both the forward and backward versions.

        Returns
        -------
            a unique tuple of 3-tuples (compound_id, phase, coefficient)

        """

        if len(sparse_with_phases) == 0:
            return tuple()

        # sort according to compound ID and normalize
        sorted_compound_list = sorted(
            sparse_with_phases.keys(), key=lambda c: c.id
        )

        # normalize the stoichiometric coefficients such that the highest
        # absolute value is 1
        max_coeff = max(map(abs, sparse_with_phases.values()))
        if max_coeff == 0:
            raise Exception("All stoichiometric ceofficients are 0")

        # if the reversible flag is on, make sure that the normalized
        # coefficient of the first compound (i.e. the one with the lowest ID)
        # is positive.
        if reversible and sparse_with_phases[sorted_compound_list[0]] < 0:
            max_coeff *= -1.0

        norm_factor = 1.0 / max_coeff
        r_list = [
            (cpd.id, cpd.phase, float(norm_factor * sparse_with_phases[cpd]))
            for cpd in sorted_compound_list
        ]
        return tuple(r_list)

    def __hash__(self) -> int:
        """Return a hash of the PhasedReaction.

        Note that the standard Python library `hash()` includes randomization
        that is initialized each time the Python kernel starts. Therefore
        this hash will not be consistent between runs. For quick reaction
        lookup, use `PhasedReaction.hash_md5()`.

        Returns
        -------
        str
            a hash of the Reaction.
        """
        if self.is_empty():
            return 0
        reactants = PhasedReaction._hashable_reactants(self.sparse_with_phases)
        return hash(reactants)

    def hash_md5(self, reversible: bool = True) -> str:
        """Return a MD5 hash of the PhasedReaction.

        This hash is useful for finding reactions with the exact same
        stoichiometry. We create a unique formula string based on the
        Compound IDs and coefficients.

        Parameters
        ----------
        reversible : bool
            a flag indicating whether the directionality of the reaction
            matters or not. if `True`, the same value will be returned for
            both the forward and backward versions.

        Returns
        -------
        hash : str
            a unique hash string representing the Reaction.
        """
        if self.is_empty():
            return ""
        reactants = PhasedReaction._hashable_reactants(
            self.sparse_with_phases, reversible=reversible
        )

        # convert the unique tuple to bytes, since hashlib requires it.
        reactants_str = str(reactants).encode()
        return hashlib.md5(reactants_str).hexdigest()

    def __eq__(self, other: "PhasedReaction") -> bool:
        """Check if two reactions are equal.

        Ignores differences in reaction direction, or a multiplication of all
        coefficients by a scalar.
        """
        reactants = PhasedReaction._hashable_reactants(self.sparse_with_phases)
        other_reactants = PhasedReaction._hashable_reactants(
            other.sparse_with_phases
        )
        return reactants == other_reactants

    def __lt__(self, other: "PhasedReaction") -> bool:
        """Create an ordering of reactions.

        Used mainly for using reactions as dictionary keys.
        """
        reactants = PhasedReaction._hashable_reactants(self.sparse_with_phases)
        other_reactants = PhasedReaction._hashable_reactants(
            other.sparse_with_phases
        )
        return reactants < other_reactants

    def set_abundance(self, compound: Compound, abundance: ureg.Quantity):
        """Set the abundance of the compound."""
        for phased_compound in self.sparse_with_phases.keys():
            if phased_compound.compound == compound:
                phased_compound.abundance = abundance

    def reset_abundances(self):
        """Reset the abundance to standard levels."""
        for phased_compound in self.sparse_with_phases.keys():
            phased_compound.reset_abundance()

    def set_phase(self, compound: Compound, phase: str):
        """Set the phase of the compound."""
        for phased_compound in self.sparse_with_phases.keys():
            if phased_compound.compound == compound:
                phased_compound.phase = phase

    def get_phased_compound(
        self, compound: Compound
    ) -> Tuple[PhasedCompound, float]:
        """Get the PhasedCompound object by the Compound object."""
        # TODO: This is not ideal. We should try to not keep two dictionaries (
        #  one with Compounds and one with PhasedCompounds).
        for phased_compound, coeff in self.sparse_with_phases.items():
            if phased_compound.compound == compound:
                return phased_compound, coeff
        return None, 0

    def get_phase(self, compound: Compound) -> str:
        """Get the phase of the compound."""
        phased_compound, _ = self.get_phased_compound(compound)
        if phased_compound is None:
            return ""
        else:
            return phased_compound.phase

    def get_abundance(self, compound: Compound) -> ureg.Quantity:
        """Get the abundance of the compound."""
        phased_compound, _ = self.get_phased_compound(compound)
        if phased_compound is None:
            return None
        else:
            return phased_compound.abundance

    @property
    def is_physiological(self) -> bool:
        """Check if all concentrations are physiological.

        This function is used by eQuilibrator to know if to present the
        adjusted dG' or not (since the physiological dG' is always
        displayed and it would be redundant).

        :return: True if all compounds are at physiological abundances.
        """
        for phased_compound in self.sparse_with_phases.keys():
            if not phased_compound.is_physiological:
                return False
        return True

    def get_stoichiometry(self, compound: Compound) -> float:
        """Get the abundance of the compound."""
        for phased_compound, coeff in self.sparse_with_phases.items():
            if phased_compound.compound == compound:
                return coeff
        return 0.0

    def add_stoichiometry(self, compound: Compound, coeff: float) -> None:
        """Add to the stoichiometric coefficient of a compound.

        If this compound is not already in the reaction, add it.
        """
        super(PhasedReaction, self).add_stoichiometry(compound, coeff)

        for phased_compound in self.sparse_with_phases.keys():
            if phased_compound.compound == compound:
                if self.sparse_with_phases[phased_compound] == -coeff:
                    self.sparse_with_phases.pop(phased_compound)
                else:
                    self.sparse_with_phases[phased_compound] += coeff
                return

        self.sparse_with_phases[PhasedCompound(compound)] = coeff

    @ureg.check(None, None, "[concentration]", "[temperature]", "")
    def separate_stored_dg_prime(
        self,
        p_h: ureg.Quantity,
        ionic_strength: ureg.Quantity,
        temperature: ureg.Quantity,
        p_mg: ureg.Quantity,
    ) -> Tuple[Reaction, ureg.Quantity]:
        """Split the PhasedReaction to aqueous phase and all the rest.

        :param p_h:
        :param ionic_strength:
        :param temperature:
        :param p_mg:
        :return: a tuple (residual_reaction, stored_dg_prime) where
        residual_reaction is a Reaction object (excluding the compounds that
        had stored values), and stored_dg_prime is the total dG' of the
        compounds with stored values (in kJ/mol).
        """
        stored_dg_prime = 0.0
        sparse = {}  # all compounds without stored dgf values
        for phased_compound, coeff in self.sparse_with_phases.items():
            dgf_prime = phased_compound.get_stored_standard_dgf_prime(
                p_h=p_h,
                ionic_strength=ionic_strength,
                temperature=temperature,
                p_mg=p_mg,
            )

            if dgf_prime is None:
                sparse[phased_compound.compound] = coeff
            else:
                stored_dg_prime += coeff * dgf_prime

        return Reaction(sparse, arrow=self.arrow, rid=self.rid), stored_dg_prime

    def separate_stored_dg(self) -> Tuple[Reaction, ureg.Quantity]:
        """Split the PhasedReaction to aqueous phase and all the rest.

        :return: a tuple (residual_reaction, stored_dg) where
        residual_reaction is a Reaction object (excluding the compounds that
        had stored values), and stored_dg is the total dG of the
        compounds with stored values (in kJ/mol).
        """
        stored_dg_prime = 0.0
        sparse = {}  # all compounds without stored dgf values
        for phased_compound, coeff in self.sparse_with_phases.items():
            dgf_prime = phased_compound.get_stored_standard_dgf()

            if dgf_prime is None:
                sparse[phased_compound.compound] = coeff
            else:
                stored_dg_prime += coeff * dgf_prime

        return Reaction(sparse, arrow=self.arrow, rid=self.rid), stored_dg_prime

    def dg_correction(self) -> ureg.Quantity:
        """Calculate the concentration adjustment in the dG' of reaction.

        :return: the correction for delta G in units of RT
        """
        # here we check that each concentration is in suitable units,
        # depending on the phase of that compound
        ddg_over_rt = Q_(0.0)
        for phased_compound, coeff in self.sparse_with_phases.items():
            ddg_over_rt += coeff * phased_compound.ln_abundance
        return ddg_over_rt

    def physiological_dg_correction(self) -> ureg.Quantity:
        """Calculate the concentration adjustment in the dG' of reaction.

        Assuming all reactants are in the default physiological
        concentrations (i.e. 1 mM)

        :return: the correction for delta G in units of RT
        """
        ddg_over_rt = Q_(0.0)
        for phased_compound, coeff in self.sparse_with_phases.items():
            ddg_over_rt += coeff * phased_compound.ln_physiological_abundance
        return ddg_over_rt

    def serialize(self) -> List[dict]:
        """Return a serialized version of all the reaction thermo data."""
        list_of_compounds = []
        for phased_compound, coeff in self.sparse_with_phases.items():
            compound_dict = phased_compound.serialize()
            compound_dict["coefficient"] = coeff
            list_of_compounds.append(compound_dict)
        return list_of_compounds
