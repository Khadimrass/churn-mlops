"""Contrat de l'API : Pydantic valide chaque requête avant qu'elle n'atteigne le modèle."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
InternetOpt = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
                "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
                "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
                "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 190.0,
            }
        }
    )

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Ancienneté en mois")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetOpt
    OnlineBackup: InternetOpt
    DeviceProtection: InternetOpt
    TechSupport: InternetOpt
    StreamingTV: InternetOpt
    StreamingMovies: InternetOpt
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, le=500)
    TotalCharges: float = Field(ge=0, le=60_000)


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    risk_level: Literal["low", "medium", "high"]


class PredictionResponse(Prediction):
    model_version: str
    threshold: float


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str
    threshold: float
    n_high_risk: int


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    source: str
    threshold: float
    trained_at: str | None
    test_metrics: dict[str, float]


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    version: str
