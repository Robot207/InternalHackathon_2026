from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field

from .config import DEFAULT_CLOUD_THRESHOLD, DEFAULT_FINE_STEP


class FetchRequest(BaseModel):
    preset: str = "london"
    start_date: str | None = None
    end_date: str | None = None
    fine_step: float = Field(default=DEFAULT_FINE_STEP, gt=0.01, le=0.2)
    cloud_threshold: float = Field(default=DEFAULT_CLOUD_THRESHOLD, ge=0, le=100)
    force: bool = False

    def resolve_dates(self) -> tuple[str, str]:
        end = self.end_date
        if end is None:
            end = (dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)).isoformat()
        start = self.start_date
        if start is None:
            sd = dt.date.fromisoformat(end) - dt.timedelta(days=6)
            start = sd.isoformat()
        return start, end


class TrainRequest(BaseModel):
    model: str = "random_forest"
    split: str = "spatiotemporal"
    conserve: bool = True


class ApplyRequest(BaseModel):
    conserve: bool = True
