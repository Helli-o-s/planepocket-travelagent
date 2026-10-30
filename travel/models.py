"""Pydantic schemas shared by the graph, tools and deterministic logic."""
from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, Field

HotelTier = Literal["budget", "mid-range", "luxury"]
Area = Literal["flights", "hotel", "attractions", "weather"]
ALL_AREAS: list[str] = ["flights", "hotel", "attractions", "weather"]


class ParsedRequest(BaseModel):
    """What the LLM extracts from the user's words. Null means 'not stated'."""
    origin: str | None = None
    destination: str | None = None
    start_date: str | None = Field(None, description="YYYY-MM-DD if a year was stated, otherwise MM-DD")
    end_date: str | None = Field(None, description="same format as start_date")
    duration_days: int | None = Field(None, description="trip length in days, only if the user stated it")
    adults: int | None = None
    children: int | None = None
    budget: float | None = None
    currency: str | None = Field(None, description="ISO code such as INR; null if neither stated nor implied by a symbol like ₹")
    budget_covers: Literal["all", "transport_and_accommodation", "unstated"] = "unstated"
    hotel_tier: HotelTier | None = None
    interests: list[str] = []
    transport_pref: str | None = None
    dietary: list[str] = []
    accessibility: list[str] = Field([], description="e.g. 'limited walking', 'wheelchair'")


class RequirementPatch(BaseModel):
    """A follow-up change. Fields left null are unchanged; list fields hold the full new list."""
    origin: str | None = None
    destination: str | None = None
    start_date: str | None = Field(None, description="YYYY-MM-DD")
    end_date: str | None = Field(None, description="YYYY-MM-DD")
    adults: int | None = None
    children: int | None = None
    budget: float | None = None
    budget_covers: Literal["all", "transport_and_accommodation"] | None = None
    hotel_tier: HotelTier | None = None
    interests: list[str] | None = None
    transport_pref: str | None = None
    dietary: list[str] | None = None
    accessibility: list[str] | None = None
    affects: list[Area] = Field(description="which researched areas must be searched again")
    note: str = Field(description="one-line summary of the change, for the user")


class TripRequirements(BaseModel):
    """Validated, fully resolved requirements. Built only by logic.normalize()."""
    origin: str
    destination: str
    start_date: date
    end_date: date
    adults: int
    children: int = 0
    budget: float
    currency: str
    budget_covers: Literal["all", "transport_and_accommodation"]
    hotel_tier: HotelTier | None = None
    interests: list[str] = []
    transport_pref: str | None = None
    dietary: list[str] = []
    accessibility: list[str] = []
    assumptions: list[str] = []

    @property
    def days(self) -> int:
        return (self.end_date - self.start_date).days + 1

    @property
    def nights(self) -> int:
        return self.days - 1

    @property
    def travellers(self) -> int:
        return self.adults + self.children

    def day_date(self, n: int) -> date:
        return self.start_date + timedelta(days=n - 1)


# --- Itinerary: the LLM only picks IDs, times and words; logic computes every number. ---

class PlannedActivity(BaseModel):
    slot: Literal["morning", "afternoon", "evening"]
    attraction_id: str
    start: str = Field(description="HH:MM, 24-hour")
    rationale: str = Field(description="why this fits the travellers' interests/constraints")


class PlannedMeal(BaseModel):
    meal: Literal["breakfast", "lunch", "dinner"]
    restaurant_id: str | None = Field(None, description="from search_restaurants; null = hotel/casual")
    note: str = ""


class PlannedDay(BaseModel):
    day: int
    date: str = Field(description="YYYY-MM-DD")
    theme: str
    activities: list[PlannedActivity]
    meals: list[PlannedMeal]
    rest: str = Field(description="when and where to rest")
    tips: list[str] = Field(description="practical considerations: weather, clothing, timing")


class Itinerary(BaseModel):
    outbound_flight_id: str | None = None
    return_flight_id: str | None = None
    hotel_id: str | None = None
    days: list[PlannedDay]
    summary: str
    unmet: list[str] = Field([], description="requirements that could not be satisfied, stated honestly in words; no prices or durations")


class CostLine(BaseModel):
    category: Literal["transport", "accommodation", "activities", "meals", "local_transport", "contingency"]
    item: str
    amount: float
    source: Literal["LIVE", "MOCK", "ESTIMATE"]


class CostReport(BaseModel):
    lines: list[CostLine]
    by_category: dict[str, float]
    total: float
    covered_total: float  # the part the budget is meant to cover
    budget: float
    budget_covers: str
    within_budget: bool


class Issue(BaseModel):
    code: str
    severity: Literal["error", "warning"]
    message: str
    day: int | None = None
