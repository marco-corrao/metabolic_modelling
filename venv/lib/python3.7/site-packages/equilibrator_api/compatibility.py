# The MIT License (MIT)
#
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


"""Provide functions for compatibility with COBRA."""


import logging
from operator import attrgetter
from typing import Dict, List

from equilibrator_cache import CompoundCache
from equilibrator_cache.compatibility import map_cobra_metabolites

from equilibrator_api.phased_reaction import PhasedReaction


logger = logging.getLogger(__name__)


def map_cobra_reactions(
    cache: CompoundCache,
    reactions: List["cobra.Reaction"],  # noqa: F821
    **kwargs,
) -> Dict[str, PhasedReaction]:
    """
    Translate COBRA reactions to eQuilibrator phased reactions.

    Parameters
    ----------
    cache : equilibrator_cache.CompoundCache
    reactions : iterable of cobra.Reaction
        A list of reactions to map to equilibrator phased reactions.

    Other Parameters
    ----------------
    kwargs :
        Any further keyword arguments are passed to the underlying function
        for mapping metabolites.

    Returns
    -------
    dict
        A mapping from COBRA reaction identifiers to equilibrator phased
        reactions where such a mapping can be established.

    See Also
    --------
    equilibrator_cache.compatibility.map_cobra_metabolites

    """
    metabolites = sorted(
        {m for r in reactions for m in r.metabolites}, key=attrgetter("id")
    )
    met2cmpnd = map_cobra_metabolites(cache, metabolites, **kwargs)
    mapping = {}
    for rxn in reactions:
        try:
            stoichiometry = {
                met2cmpnd[met.id]: coef for met, coef in rxn.metabolites.items()
            }
        except KeyError:
            logger.warning("Incomplete compound stoichiometry in '%s'.", rxn.id)
            continue
        mapping[rxn.id] = PhasedReaction(
            sparse=stoichiometry,
            arrow="<=>" if rxn.reversibility else "=>",
            rid=rxn.id,
        )
    return mapping
