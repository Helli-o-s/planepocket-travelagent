"""SQLite persistence: one row per trip, rewritten after every successful plan or refinement,
so the saved requirements, itinerary and change history always match the conversation."""
import json
import sqlite3
import time

from .models import CostReport, Issue, Itinerary, ParsedRequest, TripRequirements

DB_PATH = "trips.db"


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS trips (id TEXT PRIMARY KEY, updated REAL, title TEXT, state TEXT, history TEXT)")
    return con


def save(trip_id: str, state: dict, history: list[dict]) -> None:
    req = state["req"]
    data = {"parsed": state["parsed"].model_dump(), "req": req.model_dump(mode="json"),
            "itinerary": state["itinerary"].model_dump() if state.get("itinerary") else None,
            "costs": state["costs"].model_dump() if state.get("costs") else None,
            "issues": [i.model_dump() for i in state.get("issues") or []],
            "results": state.get("results") or {}, "seen": state.get("seen") or []}
    title = f"{req.origin} → {req.destination}, {req.start_date:%d %b %Y}, {req.days} days"
    with _db() as con:
        con.execute("INSERT OR REPLACE INTO trips VALUES (?, ?, ?, ?, ?)",
                    (trip_id, time.time(), title, json.dumps(data, default=str), json.dumps(history)))


def load(trip_id: str) -> tuple[dict, list[dict]] | None:
    with _db() as con:
        row = con.execute("SELECT state, history FROM trips WHERE id = ?", (trip_id,)).fetchone()
    if not row:
        return None
    d = json.loads(row[0])
    state = {"parsed": ParsedRequest(**d["parsed"]), "req": TripRequirements(**d["req"]),
             "itinerary": Itinerary(**d["itinerary"]) if d["itinerary"] else None,
             "costs": CostReport(**d["costs"]) if d["costs"] else None,
             "issues": [Issue(**i) for i in d["issues"]], "results": d["results"], "seen": d["seen"]}
    return state, json.loads(row[1])


def list_trips() -> list[tuple[str, str, float]]:
    with _db() as con:
        return con.execute("SELECT id, title, updated FROM trips ORDER BY updated DESC").fetchall()
