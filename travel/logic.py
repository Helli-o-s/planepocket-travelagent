"""Deterministic business logic: request normalization, expenses, itinerary validation.

No LLM calls here. Every number the user sees is computed in this module from
IDs the agent selected, never taken from model output.
"""
import math
import re
from datetime import date

from .models import CostLine, CostReport, Issue, Itinerary, ParsedRequest, TripRequirements
from .tools import diet_tag, lookup, travel_estimate

MAX_DAYS = 30
CONTINGENCY = 0.10
MEAL_ESTIMATE = {"breakfast": 200, "lunch": 400, "dinner": 500}  # per person, when no restaurant picked
MEAL_TIME = {"breakfast": "08:00", "lunch": "13:00", "dinner": "20:00"}  # typical times, for flight-day costing
MAX_TRAVEL_MIN_PER_DAY = 180  # warn above this
HARD_TRAVEL_MIN_PER_DAY = 240  # reject (triggers a replan) above this
AIRPORT_BUFFER_MIN = 90       # landing -> first activity
CHECKIN_BUFFER_MIN = 120      # leave for airport this long before departure
LIMITED_WALKING = re.compile(r"walk|wheelchair|mobility|elderly|knee", re.I)


def _parse_day(s: str, today: date, after: date | None = None) -> date:
    """YYYY-MM-DD as given; MM-DD resolves to the next occurrence on/after `after` (or today)."""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return date.fromisoformat(s)
    m, d = map(int, re.fullmatch(r"(\d{1,2})-(\d{1,2})", s).groups())
    floor = after or today
    candidate = date(floor.year, m, d)
    return candidate if candidate >= floor else date(floor.year + 1, m, d)


def normalize(p: ParsedRequest, today: date) -> tuple[TripRequirements | None, list[str]]:
    """Resolve and validate parsed input. Returns (requirements, questions); exactly one is empty."""
    q: list[str] = []
    assumptions: list[str] = []
    if not p.origin:
        q.append("Where will you be travelling from?")
    if not p.destination:
        q.append("Where would you like to go?")
    if p.origin and p.destination and p.origin.strip().lower() == p.destination.strip().lower():
        q.append("The origin and destination are the same. Where would you like to go?")
    if not p.adults:
        q.append("How many adults (and children, if any) are travelling?")
    elif not 1 <= p.adults <= 9 or not 0 <= (p.children or 0) <= 9:
        q.append("Please give 1-9 adults and 0-9 children.")
    if p.budget is None:
        q.append("What is your total budget?")
    elif p.budget <= 0:
        q.append("The budget must be a positive amount. What is your budget?")
    elif not p.currency:
        q.append(f"Which currency is the budget of {p.budget:g} in?")

    start = end = None
    try:
        start = _parse_day(p.start_date, today) if p.start_date else None
        end = _parse_day(p.end_date, today, after=start) if p.end_date else None
    except (ValueError, AttributeError):
        q.append("I couldn't read those dates. Please give them like '15 December' or 2026-12-15.")
    if start and not end and p.duration_days:
        end = date.fromordinal(start.toordinal() + p.duration_days - 1)
        assumptions.append(f"End date {end:%d %b %Y} derived from {p.duration_days}-day duration.")
    if not start or not end:
        if not any("dates" in x for x in q):
            q.append("What are your travel dates (start and end)?")
    else:
        if start < today:
            q.append(f"{start:%d %b %Y} is in the past. What dates did you mean?")
        elif end < start:
            q.append("The end date is before the start date. Please check the dates.")
        elif (end - start).days + 1 > MAX_DAYS:
            q.append(f"Trips are limited to {MAX_DAYS} days. Please shorten the dates.")
        elif p.duration_days and p.duration_days != (end - start).days + 1:
            q.append(f"You said {p.duration_days} days, but {start:%d %b} to {end:%d %b} is "
                     f"{(end - start).days + 1} days. Which is right?")
        if p.start_date and not re.match(r"\d{4}", p.start_date) and not q:
            assumptions.append(f"No year given; using the next upcoming dates ({start:%d %b %Y}).")

    if q:
        return None, q
    covers = p.budget_covers
    if covers == "unstated":
        covers = "all"
        assumptions.append("Budget assumed to cover everything (travel, stay, food, activities).")
    return TripRequirements(
        origin=p.origin, destination=p.destination, start_date=start, end_date=end,
        adults=p.adults, children=p.children or 0, budget=p.budget, currency=p.currency.upper(),
        budget_covers=covers, hotel_tier=p.hotel_tier, interests=p.interests, transport_pref=p.transport_pref,
        dietary=p.dietary, accessibility=p.accessibility, assumptions=assumptions), []


def to_min(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m


def schedule(itin: Itinerary, day_index: int) -> list[dict]:
    """Activities of one day in time order, with duration and the leg from the previous stop."""
    hotel = lookup(itin.hotel_id, "H-") if itin.hotel_id else None
    prev = hotel["area"] if hotel else None
    out = []
    for act in sorted(itin.days[day_index].activities, key=lambda a: a.start):
        a = lookup(act.attraction_id, "A-")
        if not a:
            continue
        leg = travel_estimate(prev, a["area"]) if prev else {"minutes": 0, "cost": 0}
        start = to_min(act.start)
        out.append({"act": act, "a": a, "start": start, "end": start + round(a["duration_hr"] * 60),
                    "leg_min": leg.get("minutes", 0), "leg_cost": leg.get("cost", 0), "from": prev})
        prev = a["area"]
    return out


def calc_expenses(req: TripRequirements, itin: Itinerary) -> CostReport:
    pax = req.travellers
    lines: list[CostLine] = []
    add = lambda cat, item, amt, src: lines.append(CostLine(category=cat, item=item, amount=round(amt), source=src))

    for fid in dict.fromkeys(filter(None, [itin.outbound_flight_id, itin.return_flight_id])):
        if f := lookup(fid, "FL-"):
            add("transport", f"Flight {f['airline']} {f['from']}→{f['to']} {f['date']} {f['depart']} × {pax}",
                f["price_per_person"] * pax, "MOCK")
    out_f = lookup(itin.outbound_flight_id, "FL-") if itin.outbound_flight_id else None
    ret_f = lookup(itin.return_flight_id, "FL-") if itin.return_flight_id else None
    hotel = lookup(itin.hotel_id, "H-") if itin.hotel_id else None
    breakfast_included = bool(hotel and "breakfast" in hotel["amenities"])
    if hotel:
        rooms = math.ceil(pax / hotel["capacity"])
        add("accommodation", f"{hotel['name']} × {req.nights} nights × {rooms} room(s)",
            hotel["price_per_night"] * req.nights * rooms, "MOCK")
        if itin.outbound_flight_id:
            t = travel_estimate("Goa Airport", hotel["area"], travellers=pax)
            if t["status"] == "ok":
                add("transport", "Airport → hotel taxi", t["cost"], "ESTIMATE")
        if itin.return_flight_id:
            t = travel_estimate(hotel["area"], "Goa Airport", travellers=pax)
            if t["status"] == "ok":
                add("transport", "Hotel → airport taxi", t["cost"], "ESTIMATE")

    paid = set()  # an attraction is paid for once even if the plan lists it twice
    for i, day in enumerate(itin.days):
        stops = schedule(itin, i)
        for s in stops:
            if s["a"]["id"] not in paid and s["a"]["cost_per_person"]:
                paid.add(s["a"]["id"])
                add("activities", f"Day {day.day}: {s['a']['name']} × {pax}", s["a"]["cost_per_person"] * pax, "MOCK")
            if s["leg_cost"]:
                add("local_transport", f"Day {day.day}: taxi {s['from']} → {s['a']['area']}", s["leg_cost"], "ESTIMATE")
        if stops and hotel:
            back = travel_estimate(stops[-1]["a"]["area"], hotel["area"], travellers=pax)
            if back.get("cost"):
                add("local_transport", f"Day {day.day}: taxi back to hotel", back["cost"], "ESTIMATE")
        for m in day.meals:
            t = to_min(MEAL_TIME[m.meal])
            if i == 0 and out_f and t < to_min(out_f["arrive"]):
                continue  # before landing: not spent at the destination
            if i == len(itin.days) - 1 and ret_f and t > to_min(ret_f["depart"]):
                continue  # after the return flight: not a trip expense
            if m.meal == "breakfast" and breakfast_included and i > 0:
                continue  # already in the room rate (only after a night's stay)
            r = lookup(m.restaurant_id, "R-") if m.restaurant_id else None
            if r:
                add("meals", f"Day {day.day} {m.meal}: {r['name']} × {pax}", r["avg_cost_per_person"] * pax, "MOCK")
            else:
                add("meals", f"Day {day.day} {m.meal} (estimate) × {pax}", MEAL_ESTIMATE[m.meal] * pax, "ESTIMATE")

    subtotal = sum(l.amount for l in lines)
    add("contingency", f"{CONTINGENCY:.0%} contingency", subtotal * CONTINGENCY, "ESTIMATE")
    by_cat: dict[str, float] = {}
    for l in lines:
        by_cat[l.category] = by_cat.get(l.category, 0) + l.amount
    covered_cats = {"transport", "accommodation"} if req.budget_covers == "transport_and_accommodation" else set(by_cat) - {"contingency"}
    covered = sum(v for k, v in by_cat.items() if k in covered_cats)
    return CostReport(lines=lines, by_category=by_cat, total=sum(by_cat.values()), covered_total=covered,
                      budget=req.budget, budget_covers=req.budget_covers, within_budget=covered <= req.budget)


def validate(req: TripRequirements, itin: Itinerary, costs: CostReport, seen: set[str]) -> list[Issue]:
    issues: list[Issue] = []
    err = lambda code, msg, day=None: issues.append(Issue(code=code, severity="error", message=msg, day=day))
    warn = lambda code, msg, day=None: issues.append(Issue(code=code, severity="warning", message=msg, day=day))

    # Every ID must exist and must have come from a tool result in this session.
    ids = [(itin.outbound_flight_id, "FL-"), (itin.return_flight_id, "FL-"), (itin.hotel_id, "H-")]
    ids += [(a.attraction_id, "A-") for d in itin.days for a in d.activities]
    ids += [(m.restaurant_id, "R-") for d in itin.days for m in d.meals]
    for i, kind in ids:
        if not i:
            continue
        if not lookup(i):
            err("unknown_id", f"{i} does not exist in any search result.")
        elif not lookup(i, kind):
            err("wrong_kind", f"{i} is in a slot that needs a {kind}… ID.")
        elif i not in seen:
            err("unsearched_id", f"{i} was used without being returned by a search tool.")

    if len(itin.days) != req.days:
        err("day_count", f"Plan has {len(itin.days)} days but the trip is {req.days} days.")
    for n, d in enumerate(itin.days, 1):
        if d.date != req.day_date(n).isoformat():
            err("day_date", f"Day {n} should be {req.day_date(n)}, not {d.date}.", n)

    out_f = lookup(itin.outbound_flight_id, "FL-") if itin.outbound_flight_id else None
    ret_f = lookup(itin.return_flight_id, "FL-") if itin.return_flight_id else None
    if out_f and out_f["date"] != req.start_date.isoformat():
        err("flight_date", f"Outbound flight is on {out_f['date']}, trip starts {req.start_date}.")
    if ret_f and ret_f["date"] != req.end_date.isoformat():
        err("flight_date", f"Return flight is on {ret_f['date']}, trip ends {req.end_date}.")
    if not out_f or not ret_f:
        warn("no_flights", "Round-trip flights are not both included; intercity transport cost is incomplete.")

    hotel = lookup(itin.hotel_id, "H-") if itin.hotel_id else None
    if not hotel:
        warn("no_hotel", "No hotel selected; accommodation cost is missing.")
    elif req.hotel_tier and hotel["tier"] != req.hotel_tier:
        warn("hotel_tier", f"{hotel['name']} is {hotel['tier']}, you asked for {req.hotel_tier}.")
    if hotel and req.children and not hotel["kid_friendly"]:
        err("hotel_kids", f"{hotel['name']} is not marked child-friendly.")

    limited_walking = any(LIMITED_WALKING.search(a) for a in req.accessibility)
    diets = {diet_tag(d) for d in req.dietary} - {None}
    visited: set[str] = set()
    for i, day in enumerate(itin.days):
        n, weekday = day.day, req.day_date(i + 1).weekday()
        stops, travel = schedule(itin, i), 0
        for k, s in enumerate(stops):
            a, act = s["a"], s["act"]
            travel += s["leg_min"]
            if a["id"] in visited:
                err("duplicate", f"{a['name']} appears more than once.", n)
            visited.add(a["id"])
            if weekday not in a["open_days"]:
                err("closed_day", f"{a['name']} is closed on {req.day_date(i + 1):%A}.", n)
            if s["start"] < to_min(a["open"]) or s["end"] > to_min(a["close"]):
                err("hours", f"{a['name']} at {act.start} falls outside opening hours {a['open']}-{a['close']}.", n)
            if k and stops[k - 1]["end"] + s["leg_min"] > s["start"]:
                err("overlap", f"{a['name']} at {act.start} overlaps the previous stop "
                               f"(it ends {stops[k - 1]['end'] // 60:02d}:{stops[k - 1]['end'] % 60:02d} "
                               f"+ {s['leg_min']} min travel).", n)
            if req.children and not a["kid_friendly"]:
                err("kids", f"{a['name']} is not suitable for children.", n)
            if limited_walking and a["walking"] == "high":
                err("walking", f"{a['name']} involves extensive walking.", n)
        if stops and hotel:
            travel += travel_estimate(stops[-1]["a"]["area"], hotel["area"]).get("minutes", 0)  # ride back
        if travel > HARD_TRAVEL_MIN_PER_DAY:
            err("travel_time", f"{travel} min of travel on day {n}; pick a more central hotel or group nearby places.", n)
        elif travel > MAX_TRAVEL_MIN_PER_DAY:
            warn("travel_time", f"{travel} min of travel on day {n}; consider grouping nearby places.", n)
        if stops and n == 1 and out_f and stops[0]["start"] < to_min(out_f["arrive"]) + AIRPORT_BUFFER_MIN:
            ready = to_min(out_f["arrive"]) + AIRPORT_BUFFER_MIN
            err("arrival", f"Day 1 starts at {stops[0]['act'].start} but the flight lands {out_f['arrive']}; "
                           f"allow {AIRPORT_BUFFER_MIN} min to reach the city (earliest {ready // 60:02d}:{ready % 60:02d}).", n)
        if stops and n == len(itin.days) and ret_f and stops[-1]["end"] + CHECKIN_BUFFER_MIN > to_min(ret_f["depart"]):
            err("departure", f"Last activity ends too close to the {ret_f['depart']} return flight.", n)
        for m in day.meals:
            r = lookup(m.restaurant_id, "R-") if m.restaurant_id else None
            if r and diets and not diets <= set(r.get("diet_tags", [])):
                err("diet", f"{r['name']} does not serve {', '.join(sorted(diets))} food.", n)

    ate_out = any(lookup(m.restaurant_id, "R-") for d in itin.days for m in d.meals if m.restaurant_id)
    for interest in req.interests:
        w = interest.lower()
        if ate_out and re.search(r"food|cuisine|eat|dining", w):
            continue  # food interests are met by the restaurants, not the attractions
        if not any(any(t in w or w in t for t in lookup(x)["tags"]) for x in visited):
            warn("interest", f"Nothing in the plan matches the interest '{interest}'.")

    if not costs.within_budget:
        scope = "transport + accommodation" if req.budget_covers == "transport_and_accommodation" else "the trip"
        err("budget", f"{scope.capitalize()} costs {costs.covered_total:,.0f} {req.currency}, over the "
                      f"{req.budget:,.0f} budget by {costs.covered_total - req.budget:,.0f}.")
    return issues
