from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class StatusResponse(BaseModel):
    project: str
    team: str
    phase: str
    module: str
    api: str
    dsp_core: str
    features: list[str]
