# Project Rules

## Project Type

This repository implements a Master's thesis prototype.

Primary objective:

Demonstrate that explainability can be operationalized as an independent service capability through standardized interfaces and model-agnostic orchestration.

---

## This Project IS

- a service-oriented architecture demonstration

- a proof-of-concept prototype

- an API-driven explainability service

- an architectural research artifact

---

## This Project IS NOT

- an MLOps platform

- a production-grade cloud platform

- a frontend application

- a benchmarking framework

- a novel explainability algorithm

- an explainability evaluation study

---

## Do Not Add Unless Explicitly Required

- databases

- authentication systems

- user management

- React or frontend frameworks

- Docker orchestration

- Kubernetes

- message queues

- Redis

- Celery

- event-driven architectures

- unnecessary abstractions

- additional machine learning models

- additional explainability methods

---

## Implementation Philosophy

For every implementation decision ask:

"What is the simplest implementation that still proves the architectural claim?"

Prioritize:

- readability

- simplicity

- working code

- architectural clarity

Avoid:

- overengineering

- premature optimization

- unnecessary dependencies

---

## Architectural Boundaries

API Layer

↓

ExplanationEngine

↓

ModelAdapter + MethodManager

These boundaries should not be violated.

---

## Non-Negotiable Requirements

- SHAP and LIME must return identical response schemas.

- The same `/explain` endpoint must work for multiple models.

- Explainability logic must not exist inside route handlers.

- HTTP logic must not exist inside services.

- The architecture must remain model-agnostic.