# API Contracts

This document defines the standardized interfaces exposed by the Explainability-as-a-Service (EaaS) prototype.

The API contracts are intentionally simple and stable to demonstrate service abstraction and interface standardization.

---

# POST /predict

## Query Parameters

model:

- random_forest

- logistic_regression

---

## Request Body

```json

{

  "features": [

    value1,

    value2,

    ...,

    value30

  ]

}

```

Rules:

- exactly 30 numerical values

- values must be ordered according to the Breast Cancer dataset features

---

## Response

```json

{

  "model": "random_forest",

  "prediction": 1,

  "probability": 0.87

}

```

Fields:

- model: model used for prediction

- prediction: predicted class

- probability: probability of predicted class

---

# POST /explain

## Query Parameters

model:

- random_forest

- logistic_regression

method:

- shap

- lime

---

## Request Body

```json

{

  "features": [

    value1,

    value2,

    ...,

    value30

  ]

}

```

---

## Response

```json

{

  "model": "random_forest",

  "method": "shap",

  "prediction": 1,

  "feature_importance": {

    "mean radius": 0.142,

    "mean texture": -0.031

  }

}

```

Fields:

- model: model used

- method: explanation method used

- prediction: predicted class

- feature_importance: feature contribution values

---

# Standardization Requirement

The response schema above is mandatory.

SHAP and LIME must return exactly the same structure.

This standardized contract is one of the core architectural contributions of the thesis.

---

# Error Response

```json

{

  "error": {

    "type": "InvalidModelError",

    "message": "Unsupported model: svm"

  }

}

```

Possible Errors:

- InvalidModelError

- InvalidMethodError

- ValidationError

- InternalServerError

All endpoints must return errors using this schema.



# Version

Current API Version: v1

The API contract should remain stable throughout implementation unless a change is required to preserve the research objectives.