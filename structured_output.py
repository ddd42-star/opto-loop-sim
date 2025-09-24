from typing import List, Dict, Optional
from pydantic import BaseModel, Field, ValidationError, validator


# -----------------------------
# 1. Structured Request
# -----------------------------
class StructuredRequest(BaseModel):
    intent: str = Field(..., description="Type of microscope command")
    parameters: Dict = Field(..., description="Parameters for the command")
    explanation: str = Field(..., description="Explanation of why this command is needed")

    @validator("intent")
    def validate_intent(cls, v):
        allowed = ["move_stage", "capture_image", "set_light", "analyze_image"]
        if v not in allowed:
            raise ValueError(f"intent must be one of {allowed}")
        return v


# -----------------------------
# 2. Planner Output
# -----------------------------
class Step(BaseModel):
    action: str
    params: Dict

    @validator("action")
    def validate_action(cls, v):
        allowed = ["move_stage", "capture_image", "set_light", "analyze_image"]
        if v not in allowed:
            raise ValueError(f"action must be one of {allowed}")
        return v


class PlannerOutput(BaseModel):
    steps: List[Step]


# -----------------------------
# 3. Code Generator Output
# -----------------------------
class ActionMapping(BaseModel):
    action: str
    status: str

    @validator("status")
    def validate_status(cls, v):
        allowed = ["mapped", "unmapped"]
        if v not in allowed:
            raise ValueError(f"status must be one of {allowed}")
        return v


class CodeGeneratorOutput(BaseModel):
    code: str
    language: str
    actions_mapped: List[ActionMapping]

    @validator("language")
    def validate_language(cls, v):
        if v != "python":
            raise ValueError("Only 'python' is supported")
        return v


# -----------------------------
# 4. Executor Result
# -----------------------------
class ExecutorOutputs(BaseModel):
    images: Optional[List[str]] = []
    metadata: Optional[Dict] = {}


class ExecutorResult(BaseModel):
    status: str
    executed_code: List[str]
    outputs: ExecutorOutputs
    errors: List[str]

    @validator("status")
    def validate_status(cls, v):
        allowed = ["success", "failure"]
        if v not in allowed:
            raise ValueError(f"status must be one of {allowed}")
        return v


# -----------------------------
# 5. Summarizer Output
# -----------------------------
class SummarizerOutput(BaseModel):
    summary: str
    links: Optional[List[str]] = []
    status: str

    @validator("status")
    def validate_status(cls, v):
        allowed = ["success", "failure"]
        if v not in allowed:
            raise ValueError(f"status must be one of {allowed}")
        return v
