"""A parser for reaction formulae."""
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

from typing import Any, List, Tuple

import pyparsing


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


def _parsed_compound(c_list: List[Any]) -> Tuple[float, str]:
    """Convert a list of ."""
    if len(c_list) == 2:
        return c_list[0], c_list[1]
    elif len(c_list) == 1:
        return 1.0, c_list[0]
    else:
        raise ValueError(
            f"Error while parsing this compound field: {str(c_list)}"
        )


def _make_reaction_side_parser() -> pyparsing.Forward:
    """Build a parser for one side of a reaction.

    Coefficients are usually integral, but they can be floats or fractions too.

    Returns
    -------
    parser : pyparsing.Forward

    """
    #
    int_coeff = pyparsing.Word(pyparsing.nums)
    float_coeff = pyparsing.Word(pyparsing.nums + "." + pyparsing.nums)
    frac_coeff = int_coeff + "/" + int_coeff
    int_coeff.setParseAction(lambda i: int(i[0]))
    float_coeff.setParseAction(lambda t: float(t[0]))
    frac_coeff.setParseAction(lambda f: float(f[0]) / float(f[2]))

    coeff = pyparsing.Or([int_coeff, float_coeff, frac_coeff])
    optional_coeff = pyparsing.Optional(coeff)

    compound_separator = pyparsing.Literal("+").suppress()

    compound_name_component = pyparsing.Word(
        pyparsing.alphanums + "()", pyparsing.alphanums + "-+,()'_"
    )
    compound_name = pyparsing.Forward()
    compound_name << (
        compound_name_component + pyparsing.ZeroOrMore(compound_name_component)
    )
    compound_name.setParseAction(lambda s: " ".join(s))

    compound_with_coeff = pyparsing.Forward()
    compound_with_coeff << ((optional_coeff + compound_name) | compound_name)
    compound_with_coeff.setParseAction(_parsed_compound)
    compound_with_coeff.setResultsName("compound")

    compound_with_separator = pyparsing.Forward()
    compound_with_separator << (compound_with_coeff + compound_separator)

    reaction_side = pyparsing.Forward()
    reaction_side << (
        pyparsing.ZeroOrMore(compound_with_separator) + compound_with_coeff
    )
    reaction_side.setParseAction(lambda l: [l])
    reaction_side.setResultsName("reaction_side")
    return reaction_side


def make_reaction_parser() -> pyparsing.Forward:
    """Build pyparsing-based recursive descent parser for chemical reactions.

    Returns
    -------
    parser : pyparsing.Forward

    """
    reaction_side = _make_reaction_side_parser()

    side_separators = [pyparsing.Literal(s) for s in POSSIBLE_REACTION_ARROWS]
    side_separator = pyparsing.Or(side_separators)

    reaction = pyparsing.Forward()
    reaction << (reaction_side + side_separator + reaction_side)
    return reaction
