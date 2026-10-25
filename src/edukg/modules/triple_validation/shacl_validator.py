from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from edukg.core.models import Entity, Triple, ValidationIssue, ValidationStatus
from edukg.core.models.ontology import OntologyRelation
from edukg.modules.ontology.active_ontology import ActiveOntology

if TYPE_CHECKING:
    from rdflib import Graph, URIRef

_NS = "urn:edukg:"


def _class_uri(class_id: UUID) -> URIRef:
    from rdflib import URIRef

    return URIRef(f"{_NS}class:{class_id}")


def _relation_uri(name: str) -> URIRef:
    from rdflib import URIRef

    return URIRef(f"{_NS}relation:{name}")


def _entity_uri(entity_id: UUID) -> URIRef:
    from rdflib import URIRef

    return URIRef(f"{_NS}entity:{entity_id}")


def _build_shapes_graph(relation: OntologyRelation) -> Graph:
    """One SHACL NodeShape expressing this single relation's domain/range
    constraint. Targets via `sh:targetSubjectsOf` (every node with an
    outgoing edge for this relation), not `sh:targetClass` — targetClass
    would only select nodes *already* typed as the domain class, so a
    subject typed wrong (exactly the violation this exists to catch) would
    never be selected as a target at all and silently conform. With
    targetSubjectsOf, the shape always applies to this triple's subject,
    and a direct `sh:class` constraint on the shape itself (not nested in a
    property shape — valid SHACL Core, `sh:class` applies to a shape's own
    focus nodes) checks whether that subject's own type matches
    domain_class — this is the actual domain check. The nested property
    shape below is the range check: values of this relation's property
    (the objects) must be typed as range_class. Built per-relation (not
    once for the whole ontology) to keep validate_triple() below simple."""
    from rdflib import RDF, BNode, Graph, Namespace

    sh = Namespace("http://www.w3.org/ns/shacl#")
    g = Graph()
    g.bind("sh", sh)

    shape = _relation_uri(f"shape:{relation.name}")
    g.add((shape, RDF.type, sh.NodeShape))
    g.add((shape, sh.targetSubjectsOf, _relation_uri(relation.name)))
    g.add((shape, sh["class"], _class_uri(relation.domain_class_id)))

    prop = BNode()
    g.add((shape, sh.property, prop))
    g.add((prop, sh.path, _relation_uri(relation.name)))
    g.add((prop, sh["class"], _class_uri(relation.range_class_id)))
    return g


def _build_data_graph(triple: Triple, subject: Entity, obj: Entity, relation: OntologyRelation) -> Graph:
    from rdflib import RDF, Graph

    # validate_triple's guard clause already ensures both are non-None before
    # calling here; asserted explicitly so this function is self-defending
    # (not just relying on the caller) and so mypy can narrow UUID | None.
    assert subject.ontology_class_id is not None
    assert obj.ontology_class_id is not None

    g = Graph()
    g.add((_entity_uri(subject.id), RDF.type, _class_uri(subject.ontology_class_id)))
    g.add((_entity_uri(obj.id), RDF.type, _class_uri(obj.ontology_class_id)))
    g.add((_entity_uri(subject.id), _relation_uri(relation.name), _entity_uri(obj.id)))
    return g


def validate_triple(
    triple: Triple, subject: Entity | None, obj: Entity | None, active_ontology: ActiveOntology
) -> list[ValidationIssue]:
    """
    Real pyshacl-backed domain/range validation (the PDF spec's literal
    "validates triples using pyshacl" requirement — see
    modules/triple_validation/domain_range_validator.py's docstring for why
    a rule-based checker existed first: pyshacl/rdflib weren't installed,
    and most entities resolve to the generic "Concept" ontology class since
    typing is coarse, which is still true here and still why results are
    WARNING, not INVALID — this doesn't change that judgment call, it
    replaces *how* the same check is performed).

    Builds a fresh single-relation shapes graph and single-triple data graph
    per call rather than one whole-ontology/whole-batch graph: this mirrors
    TripleValidationService.validate()'s existing per-triple contract
    exactly, so a pyshacl violation never needs to be matched back to which
    of several triples it came from (there's only ever one in the graph).
    Triple validation runs during pipeline batch processing, not a hot API
    path, so the per-call graph-construction cost is an acceptable trade for
    that simplicity — the same kind of scoped, documented trade-off already
    used throughout this module.
    """
    if subject is None or obj is None or subject.ontology_class_id is None or obj.ontology_class_id is None:
        return []  # nothing resolvable to check — the rule-based checker already reports this case
    relation = active_ontology.get_relation(triple.predicate)
    if relation is None:
        return []  # UNKNOWN_PREDICATE is already reported by the rule-based checker

    from pyshacl import validate as pyshacl_validate

    shapes_graph = _build_shapes_graph(relation)
    data_graph = _build_data_graph(triple, subject, obj, relation)
    conforms, _results_graph, results_text = pyshacl_validate(data_graph, shacl_graph=shapes_graph, inference=None)

    if conforms:
        return []
    return [
        ValidationIssue(
            code="SHACL_DOMAIN_RANGE_VIOLATION",
            message=f"pyshacl: {results_text.strip()}",
            severity=ValidationStatus.WARNING,
            field_path="predicate",
        )
    ]
