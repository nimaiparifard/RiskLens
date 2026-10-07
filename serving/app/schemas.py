"""Pydantic request / response models (typed boundary of the model server)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CreditFeatures(BaseModel):
    """Raw application fields (Home Credit names). Unknown fields are allowed
    and forwarded - e.g. pre-computed bureau / previous-loan aggregates
    (``BURO_*``, ``PREV_*``, ``INSTAL_*`` ...). Missing values are fine: the
    GBDT models handle NaN natively."""

    model_config = ConfigDict(extra="allow")

    SK_ID_CURR: int | None = None
    NAME_CONTRACT_TYPE: Literal["Cash loans", "Revolving loans"] | None = "Cash loans"
    CODE_GENDER: str | None = None
    FLAG_OWN_CAR: Literal["Y", "N"] | None = None
    FLAG_OWN_REALTY: Literal["Y", "N"] | None = None
    CNT_CHILDREN: int | None = Field(None, ge=0)
    CNT_FAM_MEMBERS: float | None = Field(None, ge=0)
    AMT_INCOME_TOTAL: float = Field(..., gt=0)
    AMT_CREDIT: float = Field(..., gt=0)
    AMT_ANNUITY: float | None = Field(None, gt=0)
    AMT_GOODS_PRICE: float | None = Field(None, ge=0)
    DAYS_BIRTH: int = Field(..., lt=0, description="negative days relative to application")
    DAYS_EMPLOYED: int | None = None
    DAYS_REGISTRATION: float | None = None
    DAYS_ID_PUBLISH: int | None = None
    EXT_SOURCE_1: float | None = Field(None, ge=0, le=1)
    EXT_SOURCE_2: float | None = Field(None, ge=0, le=1)
    EXT_SOURCE_3: float | None = Field(None, ge=0, le=1)
    NAME_INCOME_TYPE: str | None = None
    NAME_EDUCATION_TYPE: str | None = None
    NAME_FAMILY_STATUS: str | None = None
    NAME_HOUSING_TYPE: str | None = None
    OCCUPATION_TYPE: str | None = None
    ORGANIZATION_TYPE: str | None = None
    REGION_RATING_CLIENT: int | None = None


class CreditScoreRequest(BaseModel):
    features: CreditFeatures
    explain: bool = True


class TransactionFeatures(BaseModel):
    """Raw IEEE-CIS style transaction. Extra fields (C*, D*, M*, V*, id_*) allowed."""

    model_config = ConfigDict(extra="allow")

    TransactionID: int
    TransactionDT: int = Field(..., ge=0, description="seconds since reference time")
    TransactionAmt: float = Field(..., gt=0)
    ProductCD: Literal["W", "C", "R", "H", "S"] = "W"
    card1: int
    card2: float | None = None
    card3: float | None = None
    card4: str | None = None
    card5: float | None = None
    card6: str | None = None
    addr1: float | None = None
    addr2: float | None = None
    dist1: float | None = None
    P_emaildomain: str | None = None
    R_emaildomain: str | None = None
    DeviceType: str | None = None
    DeviceInfo: str | None = None
    D1: float | None = None


class FraudScoreRequest(BaseModel):
    transaction: TransactionFeatures
    explain: bool | Literal["flagged"] = Field(
        "flagged",
        description='SHAP reasons: true = always, "flagged" = only review/block (fast path)',
    )
    record_history: bool = Field(
        True, description="append to the card's online history after scoring"
    )


class ReasonOut(BaseModel):
    feature: str
    value: Any = None
    contribution: float
    direction: Literal["increases_risk", "decreases_risk"]
    text: str


class ShadowResult(BaseModel):
    model_version: str
    probability: float
    decision: str


class ScoreResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    task: Literal["credit", "fraud"]
    probability: float = Field(..., description="calibrated probability of default / fraud")
    score_points: int | None = Field(
        None, description="scorecard points (credit only, higher = safer)"
    )
    grade: str
    decision: str
    threshold: float
    reasons: list[ReasonOut] = []
    model_name: str
    model_version: str
    latency_ms: float
    request_id: str | None = None
    shadow: ShadowResult | None = None


class BatchCreditRequest(BaseModel):
    items: list[CreditFeatures] = Field(..., min_length=1, max_length=5000)
    explain: bool = True


class BatchResponse(BaseModel):
    results: list[ScoreResponse]


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    task: str
    loaded: bool
    model_name: str
    version: str | None = None
    source: str | None = None
    loaded_at: str | None = None
    threshold: float | None = None
    metrics: dict[str, float] = {}


class DriftRequest(BaseModel):
    task: Literal["credit", "fraud"] = "credit"
    records: list[dict[str, Any]] = Field(..., min_length=1)
    scores: list[float] | None = None
    psi_threshold: float = 0.2
