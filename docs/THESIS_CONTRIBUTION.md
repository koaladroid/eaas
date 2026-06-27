# Thesis Contribution

## Core Contribution

This thesis investigates whether explainability can be operationalized as an independent service capability through a standardized, API-driven architectural abstraction.

The contribution of this work is architectural rather than algorithmic.

This work does not propose new explainability algorithms or improve existing methods such as SHAP and LIME.

Instead, it demonstrates that explainability can be externalized into a reusable service layer.

---

# Research Claims

The system must demonstrate the following claims.

## Claim 1

Explainability can be externalized into an independently deployable service.

Evidence:

- Public API deployment

- Dedicated `/explain` endpoint

---

## Claim 2

Heterogeneous explanation methods can share a standardized interface.

Evidence:

- SHAP and LIME return identical response schemas.

---

## Claim 3

Explainability can be orchestrated independently of model implementations.

Evidence:

- ExplanationEngine orchestrates the workflow.

- Model details are hidden behind ModelAdapter.

---

## Claim 4

Explainability can be model-agnostic.

Evidence:

- The same architecture supports both:

  - RandomForest

  - LogisticRegression

without architectural modifications.

---

# Litmus Test

Every implementation decision should answer:

Does this change help demonstrate one or more of the following?

- service abstraction

- interface standardization

- orchestration independence

- model agnosticism

If the answer is NO, the change should be rejected.

---

# Success Statement

The thesis is successful if it demonstrates:

"Explainability can function as an independently orchestrated service capability rather than an embedded model utility."