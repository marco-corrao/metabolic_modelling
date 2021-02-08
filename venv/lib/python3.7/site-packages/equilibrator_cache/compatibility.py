"""A module for mapping compounds from different nomenclatures."""
# The MIT License (MIT)
#
# Copyright (c) 2018 Institute for Molecular Systems Biology, ETH Zurich.
# Copyright (c) 2018, 2019 Novo Nordisk Foundation Center for Biosustainability,
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


import logging
from typing import Dict, List, Tuple

from .compound_cache import CompoundCache
from .models import Compound


__all__ = ("map_cobra_metabolites",)


logger = logging.getLogger(__name__)


AnnotationGrouping = Dict[str, List[Tuple["cobra.Metabolite", str]]]


def group_metabolites_by_annotation(
    metabolites: List["cobra.Metabolite"], annotation_order: Tuple[str, ...]
) -> AnnotationGrouping:
    """
    Group metabolites by an ordered list of annotation namespaces.

    Parameters
    ----------
    metabolites : list of cobra.Metabolite
        The COBRA metabolites of interest to be mapped.
    annotation_order : tuple of str
        The preferred order of namespaces to group metabolites by.

    Returns
    -------
    dict
        A mapping from namespaces to lists of pairs of metabolites and their
        respective matching annotation within that namespace.

    """
    grouping = {}
    for met in metabolites:
        for namespace in annotation_order:
            if namespace in met.annotation:
                # TODO: Handle list annotations properly.
                annotation = met.annotation[namespace]
                if isinstance(annotation, list):
                    annotation = annotation[0]
                grouping.setdefault(namespace, []).append((met, annotation))
                break
    return grouping


def map_metabolites_to_compounds(
    cache: CompoundCache,
    grouping: AnnotationGrouping,
    inchi_key_namespace: str,
    inchi_namespace: str,
) -> Dict[str, Compound]:
    """
    Map COBRA metabolites to compounds from the cache.

    Parameters
    ----------
    cache : equilibrator_cache.CompoundCache
        A compound cache.
    grouping : dict
        A mapping from namespaces to lists of pairs of metabolites and their
        respective matching annotation within that namespace.
    inchi_key_namespace : str
        Which annotation should be considered to describe an InChI Key.
    inchi_namespace : str
        Which annotation should be considered to describe an InChI.

    Returns
    -------
    dict
        A map from COBRA metabolite identifiers to compounds in the cache.

    """

    mapping = {}
    for ns, pairs in grouping.items():
        identifiers = {p[1] for p in pairs}
        # Create a reverse map from relevant compound identifiers in the
        # current namespace to the compound instances.
        if ns == inchi_key_namespace:
            compounds = cache.get_compounds(ns, identifiers, is_inchi_key=True)
            ann2cmpnd = {
                (ns, c.inchi_key): c
                for c in compounds
                if c.inchi_key in identifiers
            }
        elif ns == inchi_namespace:
            compounds = cache.get_compounds(ns, identifiers, is_inchi=True)
            ann2cmpnd = {
                (ns, c.inchi): c for c in compounds if c.inchi in identifiers
            }
        else:
            compounds = cache.get_compounds(ns, identifiers)
            ann2cmpnd = {
                (ns, i.accession): c
                for c in compounds
                for i in c.identifiers
                if i.registry.namespace == ns and i.accession in identifiers
            }
        # Use the generated reverse map to create a direct relationship
        # between COBRA metabolites and compounds.
        for met, ann in pairs:
            annotation = (ns, ann)
            if annotation in ann2cmpnd:
                mapping[met.id] = ann2cmpnd[(ns, ann)]
    return mapping


def map_cobra_metabolites(
    cache: CompoundCache,
    metabolites: List["cobra.Metabolite"],
    annotation_preference: Tuple[str, ...] = (
        "inchikey",
        "inchi",
        "metanetx.chemical",
        "seed.compound",
        "metacyc.compound",
        "kegg.compound",
        "kegg.drug",
        "kegg.glycan",
        "bigg.metabolite",
        "sabiork.compound",
        "chebi",
        "hmdb",
        "reactome",
        "lipidmaps",
        "envipath",
    ),
    inchi_key_namespace: str = "inchikey",
    inchi_namespace: str = "inchi",
) -> Dict[str, Compound]:
    """
    Map COBRA metabolites to compounds from the cache.

    Parameters
    ----------
    cache : equilibrator_cache.CompoundCache
        A compound cache.
    metabolites : list of cobra.Metabolite
        The COBRA metabolites of interest to be mapped.
    annotation_preference : tuple, optional
        The preferred order of namespaces to consider. This should range from
        most to least specific/trustworthy. Namespaces are expected to follow
        Identifiers.org standards.
    inchi_key_namespace : str
        Which annotation should be considered to describe an InChI Key.
    inchi_namespace : str
        Which annotation should be considered to describe an InChI.

    Returns
    -------
    dict of str to Compound
        A map from COBRA metabolite identifiers to compounds in the cache.

    """
    if len(metabolites) == 0:
        return {}
    grouping = group_metabolites_by_annotation(
        metabolites, annotation_preference
    )
    num_missing = len(metabolites) - sum(len(g) for g in grouping.values())
    percent = num_missing / len(metabolites) * 100.0
    logger.info(
        "%d/%d (%.2f%%) metabolites had no annotations within relevant "
        "namespaces.",
        num_missing,
        len(metabolites),
        percent,
    )
    mapping = map_metabolites_to_compounds(
        cache, grouping, inchi_key_namespace, inchi_namespace
    )
    percent = len(mapping) / len(metabolites) * 100.0
    logger.info(
        "Mapped %d/%d (%.2f%%) metabolites.",
        len(mapping),
        len(metabolites),
        percent,
    )
    return mapping
