from __future__ import annotations

from edukg.core.exceptions import OntologyError
from edukg.core.logging_setup import get_logger
from edukg.core.models import OntologyClass, OntologyExtension, OntologyExtensionKind, OntologyRelation
from edukg.modules.ontology.active_ontology import ActiveOntology
from edukg.modules.ontology.ontology_manager import OntologyManager

logger = get_logger(__name__)


class OntologyMergeService:
    """
    Turns an *approved* OntologyExtension into a real OntologyClass/OntologyRelation
    via OntologyManager. This is only ever called by OntologyReviewService after an
    approve/merge ReviewDecision — never directly from validation or extension
    generation. Alias-kind extensions attach the alias to the existing parent class
    rather than creating a new entity.
    """

    def __init__(self, ontology_manager: OntologyManager) -> None:
        self._manager = ontology_manager

    def merge(self, extension: OntologyExtension, active_ontology: ActiveOntology) -> OntologyClass | OntologyRelation:
        if extension.kind == OntologyExtensionKind.CLASS:
            return self._merge_class(extension)
        if extension.kind == OntologyExtensionKind.RELATION:
            return self._merge_relation(extension)
        if extension.kind == OntologyExtensionKind.ALIAS:
            return self._merge_alias(extension, active_ontology)
        if extension.kind == OntologyExtensionKind.CONSTRAINT:
            # Constraints (e.g. domain/range tightening) require a targeted schema change;
            # scope for this module is class/relation/alias merges. Recorded, not silently dropped.
            raise OntologyError(
                f"Constraint-kind extension {extension.id} requires manual ontology edit — not auto-mergeable"
            )
        raise OntologyError(f"Unknown OntologyExtension kind: {extension.kind}")

    def _merge_class(self, extension: OntologyExtension) -> OntologyClass:
        # Without this check, approving an LLM-proposed class whose name
        # happens to collide with an existing one (e.g. a core class like
        # "Concept", or one a teammate already approved) silently created a
        # second class under the same name — get_active_ontology's
        # classes_by_name dict then has two UUIDs fighting over one name key,
        # with whichever loaded last winning and the other orphaned (still
        # referenced by whatever used its id, but no longer reachable by
        # name). Found via a cross-check against a teammate's independently-
        # built codebase, which had already hit and fixed this.
        existing = self._manager.find_class(extension.proposed_name, extension.subject)
        if existing is not None:
            logger.info("ontology_class_merge_skipped_existing", extension_id=str(extension.id), name=existing.name)
            return existing
        new_class = OntologyClass(
            name=extension.proposed_name,
            subject=extension.subject,
            parent_class_id=extension.proposed_parent_class_id,
            description=extension.proposed_description,
            aliases=list(extension.proposed_aliases),
            is_core=False,
        )
        saved = self._manager.add_class(new_class)
        logger.info("ontology_class_merged", extension_id=str(extension.id), name=saved.name)
        return saved

    def _merge_relation(self, extension: OntologyExtension) -> OntologyRelation:
        if extension.proposed_domain_class_id is None or extension.proposed_range_class_id is None:
            raise OntologyError(f"Relation extension {extension.id} is missing domain/range — cannot merge")
        # Same reasoning as _merge_class above — this check was missing here too.
        existing = self._manager.find_relation(extension.proposed_name, extension.subject)
        if existing is not None:
            logger.info("ontology_relation_merge_skipped_existing", extension_id=str(extension.id), name=existing.name)
            return existing
        new_relation = OntologyRelation(
            name=extension.proposed_name,
            subject=extension.subject,
            domain_class_id=extension.proposed_domain_class_id,
            range_class_id=extension.proposed_range_class_id,
            description=extension.proposed_description,
        )
        saved = self._manager.add_relation(new_relation)
        logger.info("ontology_relation_merged", extension_id=str(extension.id), name=saved.name)
        return saved

    def _merge_alias(self, extension: OntologyExtension, active_ontology: ActiveOntology) -> OntologyClass:
        if extension.proposed_parent_class_id is None:
            raise OntologyError(f"Alias extension {extension.id} has no resolvable parent class — cannot merge")
        parent = next(
            (c for c in active_ontology.classes if c.id == extension.proposed_parent_class_id),
            None,
        )
        if parent is None:
            raise OntologyError(f"Alias extension {extension.id} parent class not found in active ontology")

        updated_aliases = list(parent.aliases)
        if extension.proposed_name not in updated_aliases:
            updated_aliases.append(extension.proposed_name)

        updated_class = parent.model_copy(update={"aliases": updated_aliases})
        saved = self._manager.add_class(updated_class)
        logger.info("ontology_alias_merged", extension_id=str(extension.id), alias=extension.proposed_name, parent=parent.name)
        return saved
