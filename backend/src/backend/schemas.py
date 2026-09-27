from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field

from .config import DEFAULT_CLOUD_THRESHOLD, DEFAULT_FINE_STEP


class FetchRequest(BaseModel):
    preset: str = "london"
    # Optional Indian-city selector. When present it wins over `preset` and the
    # dataset is cropped to that city's bounding box (see backend.cities).
    city: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    # ge= (not gt=) so the 0.01 deg / ~1 km target resolution is accepted.
    fine_step: float = Field(default=DEFAULT_FINE_STEP, ge=0.01, le=0.2)
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
