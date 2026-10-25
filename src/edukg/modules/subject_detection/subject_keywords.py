from __future__ import annotations

from edukg.core.models import Subject

# Assumption (flagged): no subject-detection methodology was specified in the attached
# project documents, so this offline, dependency-free, keyword-frequency strategy is used
# as the default SubjectDetectionStrategy implementation. It is intentionally swappable —
# see strategy.py — so it can be replaced with an embedding/classifier-based strategy later
# without touching the pipeline.
SUBJECT_KEYWORDS: dict[Subject, list[str]] = {
    Subject.MATHEMATICS: [
        "theorem", "proof", "axiom", "lemma", "corollary", "integral", "derivative",
        "matrix", "vector space", "eigenvalue", "polynomial", "calculus", "algebra",
        "topology", "differential equation", "probability distribution", "set theory",
    ],
    Subject.COMPUTER_SCIENCE: [
        "algorithm", "data structure", "time complexity", "big o", "binary tree",
        "linked list", "recursion", "sorting", "graph traversal", "compiler",
        "operating system", "database", "software engineering", "design pattern",
    ],
    Subject.MACHINE_LEARNING: [
        "gradient descent", "neural network", "loss function", "training set",
        "overfitting", "regularization", "backpropagation", "supervised learning",
        "feature vector", "hyperparameter", "cross-validation", "classifier",
    ],
    Subject.DEEP_LEARNING: [
        "convolutional neural network", "transformer", "attention mechanism",
        "recurrent neural network", "dropout", "batch normalization", "embedding layer",
        "activation function", "gpu training", "epoch", "fine-tuning",
    ],
    Subject.REINFORCEMENT_LEARNING: [
        "reward function", "policy gradient", "markov decision process", "q-learning",
        "agent", "environment", "exploration exploitation", "value function", "episode",
    ],
    Subject.PHYSICS: [
        "newton's law", "kinematics", "electromagnetic", "quantum mechanics", "thermodynamics",
        "velocity", "acceleration", "momentum", "electric field", "wave function",
    ],
    Subject.CHEMISTRY: [
        "chemical bond", "molarity", "reaction rate", "periodic table", "covalent",
        "ionic compound", "acid base", "stoichiometry", "electron configuration",
    ],
    Subject.BIOLOGY: [
        "cell membrane", "mitosis", "photosynthesis", "dna", "rna", "enzyme",
        "ecosystem", "evolution", "genome", "protein synthesis",
    ],
    Subject.ENGLISH: [
        "subject-verb agreement", "past tense", "vocabulary", "reading comprehension",
        "listening section", "toeic", "synonym", "grammar rule", "sentence structure",
        "preposition", "conjunction",
    ],
}
