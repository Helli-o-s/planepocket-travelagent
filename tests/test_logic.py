"""Deterministic logic and mock tools: no LLM, no network."""
from datetime import date

from travel import logic
from travel.models import Itinerary, ParsedRequest, PlannedActivity, PlannedDay, PlannedMeal
from travel.tools import lookup, search_attractions, search_flights, search_hotels, travel_estimate

TODAY = date(2026, 10, 30)
GOA = dict(origin="Indore", destination="Goa", start_date="12-15", end_date="12-19", adults=2,
           budget=35000, currency="INR", budget_covers="transport_and_accommodation", hotel_tier="mid-range",
           interests=["beaches", "local food", "historical attractions", "scenic places"])


def req(**over):
    r, q = logic.normalize(ParsedRequest(**(GOA | over)), TODAY)
    assert not q, q
    return r


def test_example_request_resolves_year_and_duration():
    r = req()
    assert (r.start_date, r.end_date, r.days, r.nights) == (date(2026, 12, 15), date(2026, 12, 19), 5, 4)
    assert any("No year given" in a for a in r.assumptions)


def test_missing_year_rolls_to_next_year_and_crosses_new_year():
    r = req(start_date="03-10", end_date="03-12")
    assert r.start_date == date(2027, 3, 10)
    r = req(start_date="12-30", end_date="01-02")
    assert (r.start_date, r.end_date) == (date(2026, 12, 30), date(2027, 1, 2))


def test_questions_for_missing_and_invalid_input():
    _, q = logic.normalize(ParsedRequest(destination="Goa"), TODAY)
    assert len(q) == 4  # origin, travellers, budget, dates
    _, q = logic.normalize(ParsedRequest(**GOA | {"duration_days": 3}), TODAY)
    assert "Which is right" in q[0]
    _, q = logic.normalize(ParsedRequest(**GOA | {"start_date": "2025-12-15", "end_date": "2025-12-19"}), TODAY)
    assert "in the past" in q[0]
    _, q = logic.normalize(ParsedRequest(**GOA | {"budget": -5}), TODAY)
    assert "positive" in q[0]
    _, q = logic.normalize(ParsedRequest(**GOA | {"currency": None}), TODAY)
    assert "currency" in q[0]


def test_unstated_budget_scope_is_strict_and_visible():
    r = req(budget_covers="unstated")
    assert r.budget_covers == "all" and any("cover everything" in a for a in r.assumptions)


def test_duration_without_end_date_is_derived():
    r = req(end_date=None, duration_days=5)
    assert r.end_date == date(2026, 12, 19)


def _plan(**over) -> Itinerary:
    out = search_flights.invoke({"origin": "Indore", "destination": "Goa", "date": "2026-12-15", "passengers": 2})
    ret = search_flights.invoke({"origin": "Goa", "destination": "Indore", "date": "2026-12-19", "passengers": 2})
    days = [PlannedDay(day=n, date=f"2026-12-{14 + n}", theme="t", rest="r", tips=[], activities=[], meals=[
        PlannedMeal(meal="breakfast"), PlannedMeal(meal="lunch", restaurant_id="R-GOA-01"), PlannedMeal(meal="dinner")])
        for n in range(1, 6)]
    days[1].activities = [PlannedActivity(slot="morning", attraction_id="A-GOA-07", start="09:30", rationale=""),
                          PlannedActivity(slot="afternoon", attraction_id="A-GOA-08", start="11:00", rationale="")]
    days[2].activities = [PlannedActivity(slot="morning", attraction_id="A-GOA-01", start="10:00", rationale=""),
                          PlannedActivity(slot="evening", attraction_id="A-GOA-10", start="18:00", rationale="")]
    base = dict(outbound_flight_id=out["items"][0]["id"], return_flight_id=ret["items"][3]["id"],
                hotel_id="H-GOA-04", days=days, summary="")
    return Itinerary(**base | over)


SEEN = {"A-GOA-07", "A-GOA-08", "A-GOA-01", "A-GOA-10", "H-GOA-04", "R-GOA-01"}


def _seen(itin):
    return SEEN | {itin.outbound_flight_id, itin.return_flight_id}


def test_flight_ids_round_trip_through_lookup():
    f = search_flights.invoke({"origin": "Indore", "destination": "Goa", "date": "2026-12-15", "passengers": 2})["items"][2]
    assert lookup(f["id"]) == f


def test_expenses_no_double_count_and_breakfast_included():
    r, itin = req(), _plan()
    itin.days[3].activities = [PlannedActivity(slot="evening", attraction_id="A-GOA-10", start="18:00", rationale="")]
    c = logic.calc_expenses(r, itin)
    cruise = [l for l in c.lines if "cruise" in l.item]
    assert len(cruise) == 1 and cruise[0].amount == 1000      # paid once, 500 × 2
    assert not any("breakfast" in l.item for l in c.lines)    # hotel H-GOA-04 includes breakfast
    hotel = next(l for l in c.lines if l.category == "accommodation")
    assert hotel.amount == 2800 * 4 * 1
    assert round(c.total) == round(sum(l.amount for l in c.lines))
    assert c.covered_total == c.by_category["transport"] + c.by_category["accommodation"]
    assert any(i.code == "duplicate" for i in logic.validate(r, itin, c, _seen(itin)))


def test_meals_follow_flights_and_hotel_breakfast():
    r = req()
    ret = search_flights.invoke({"origin": "Goa", "destination": "Indore", "date": "2026-12-19", "passengers": 2})["items"][2]
    itin = _plan(return_flight_id=ret["id"], hotel_id="H-GOA-05")   # 15:25 departure, no breakfast in room rate
    items = [l.item for l in logic.calc_expenses(r, itin).lines]
    assert not any(i.startswith("Day 1 breakfast") for i in items)  # outbound lands 08:07, after breakfast time
    assert any(i.startswith("Day 2 breakfast") for i in items)      # hotel has no breakfast: paid
    assert not any(i.startswith("Day 5 dinner") for i in items)     # after the 15:25 return flight
    assert any(i.startswith("Day 5 lunch") for i in items)
    itin = _plan(outbound_flight_id=None)                              # hotel with breakfast, no flight info
    items = [l.item for l in logic.calc_expenses(r, itin).lines]
    assert any(i.startswith("Day 1 breakfast") for i in items)      # no night stayed yet: paid
    assert not any(i.startswith("Day 2 breakfast") for i in items)  # included in room rate


def test_valid_plan_has_no_errors():
    r, itin = req(), _plan()
    errors = [i for i in logic.validate(r, itin, logic.calc_expenses(r, itin), _seen(itin)) if i.severity == "error"]
    assert errors == []


def test_validator_catches_constraint_violations():
    r = req(children=1, accessibility=["avoid extensive walking"], dietary=["vegetarian"], budget=5000)
    itin = _plan()
    itin.days[1].activities[1].start = "09:45"                                       # overlaps the Basilica
    itin.days[2].activities.append(PlannedActivity(slot="afternoon", attraction_id="A-GOA-14", start="13:00", rationale=""))
    itin.days[2].activities.append(PlannedActivity(slot="morning", attraction_id="A-GOA-12", start="08:00", rationale=""))
    itin.days[3].meals[1].restaurant_id = "R-GOA-06"                                   # non-veg only
    itin.days[4].activities = [PlannedActivity(slot="morning", attraction_id="A-GOA-99", start="09:00", rationale="")]
    codes = {i.code for i in logic.validate(r, itin, logic.calc_expenses(r, itin), _seen(itin))}
    assert {"overlap", "kids", "walking", "closed_day", "diet", "budget", "unknown_id", "unsearched_id"} <= codes


def test_arrival_and_departure_buffers():
    r, itin = req(), _plan()
    itin.days[0].activities = [PlannedActivity(slot="morning", attraction_id="A-GOA-11", start="07:00", rationale="")]
    itin.days[4].activities = [PlannedActivity(slot="evening", attraction_id="A-GOA-01", start="19:00", rationale="")]
    codes = {i.code for i in logic.validate(r, itin, logic.calc_expenses(r, itin), _seen(itin) | {"A-GOA-11"})}
    assert {"arrival", "departure"} <= codes


def test_tools_handle_bad_input_and_simulated_outage(monkeypatch):
    assert search_flights.invoke({"origin": "Indore", "destination": "Atlantis", "date": "2026-12-15", "passengers": 2})["status"] == "empty"
    assert search_flights.invoke({"origin": "Indore", "destination": "Goa", "date": "15/12", "passengers": 2})["status"] == "error"
    assert search_hotels.invoke({"city": "Goa", "guests": 2, "max_price_per_night": 100})["status"] == "empty"
    monkeypatch.setenv("MOCK_FAIL", "search_hotels")
    assert search_hotels.invoke({"city": "Goa", "guests": 2})["status"] == "error"
    monkeypatch.setenv("MOCK_FAIL", "flights")  # short form, as people naturally type it
    assert search_flights.invoke({"origin": "Indore", "destination": "Goa", "date": "2026-12-15", "passengers": 2})["status"] == "error"
    assert search_hotels.invoke({"city": "Goa", "guests": 2})["status"] == "ok"


def test_airport_codes_are_accepted():
    # Seen live: a fallback model passed IATA codes and got "no airport known for IDR".
    res = search_flights.invoke({"origin": "IDR", "destination": "GOI", "date": "2026-12-15", "passengers": 2})
    assert res["status"] == "ok" and res["items"][0]["id"].startswith("FL-IDR-GOI")
    assert search_hotels.invoke({"city": "GOI", "guests": 2})["status"] == "ok"
    # Seen live: "UNAVAILABLE estimate_transport: unknown place: Goa airport (GOI)".
    assert travel_estimate("Goa airport (GOI)", "Panaji")["status"] == "ok"
    assert travel_estimate("Indore International Airport", "Indore")["status"] == "ok"


def test_attraction_interest_ranking():
    items = search_attractions.invoke({"city": "Goa", "interests": ["historical attractions"]})["items"]
    assert "historical" in items[0]["tags"]


def test_wrong_kind_ids_are_errors_not_crashes():
    # Seen live: a flight ID reached code that expected a 'name' (KeyError). Any ID in any slot must be safe.
    from travel.tools import estimate_budget
    r = req()
    for hotel, flight, rest in [("A-GOA-01", "H-GOA-04", "FL-IDR-GOI-20261215-1"), ("FL-IDR-GOI-20261215-1", "R-GOA-01", "H-GOA-04")]:
        itin = _plan(hotel_id=hotel, outbound_flight_id=flight)
        itin.days[1].meals[1].restaurant_id = rest
        itin.days[2].activities[0].attraction_id = rest
        codes = [i.code for i in logic.validate(r, itin, logic.calc_expenses(r, itin), _seen(itin))]
        assert codes.count("wrong_kind") == 4, codes
    res = estimate_budget.invoke({"flight_ids": ["H-GOA-04"], "hotel_id": "FL-IDR-GOI-20261215-1", "nights": 4, "travellers": 2})
    assert res["status"] == "error" and len(res["unknown_ids"]) == 2


def test_food_interest_is_met_by_restaurants():
    # Seen live: "Nothing in the plan matches the interest 'local food'" although every lunch was at a Goan restaurant.
    r = req(interests=["local food"])
    itin = _plan()  # lunches at R-GOA-01, no food-tagged attraction needed
    warned = lambda it: any(i.code == "interest" for i in logic.validate(r, it, logic.calc_expenses(r, it), _seen(it)))
    assert not warned(itin)
    for d in itin.days:
        for m in d.meals:
            m.restaurant_id = None
    itin.days[1].activities, itin.days[2].activities = [], []
    assert warned(itin)
