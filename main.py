"""CLI entry point: python main.py ["trip request"] [--json]"""
import json
import math
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

from travel import graph, store  # noqa: E402  (needs env loaded first)
from travel.logic import schedule  # noqa: E402
from travel.tools import lookup, mock_book  # noqa: E402

console = Console()
LABEL_STYLE = {"LIVE": "green", "MOCK": "magenta", "ESTIMATE": "yellow", "CLIMATE": "cyan", "UNAVAILABLE": "red"}
LEGEND = ("[green]LIVE[/] real-time API · [cyan]CLIMATE[/] historical averages, not a forecast · "
          "[magenta]MOCK[/] simulated provider, not real prices/availability · [yellow]ESTIMATE[/] computed "
          "approximation · [red]UNAVAILABLE[/] source failed")
HELP = ("Commands: [bold]flights[/] · [bold]hotels[/] · [bold]costs[/] · [bold]plan[/] · [bold]json[/] · "
        "[bold]book[/] (mock) · [bold]quit[/] — anything else refines the trip.")


def label(src: str) -> str:
    return f"[{LABEL_STYLE.get(src, 'white')}]{src}[/]"


def money(x: float) -> str:
    return f"₹{x:,.0f}"


def delta(x: float) -> str:
    return "—" if not x else f"[green]-{money(-x)}[/]" if x < 0 else f"[red]+{money(x)}[/]"


# --- Live trace -------------------------------------------------------------------

def _summary(result: dict) -> str:
    if result.get("status") != "ok":
        return f"[red]{result.get('status')}[/]: {result.get('message', '')}"
    if "items" in result:
        return f"{len(result['items'])} results"
    if result.get("kind"):
        return result["kind"]
    if "minutes" in result:
        return f"{result['minutes']} min, {money(result['cost'])}"
    if "transport_and_accommodation_total" in result:
        return f"flights+hotel = {money(result['transport_and_accommodation_total'])}"
    return "ok"


def console_sink(kind: str, d: dict) -> None:
    if kind == "tool":
        args = ", ".join(f"{k}={v}" for k, v in d["args"].items())
        cached = " [dim](cached)[/]" if d["cached"] else ""
        console.print(f"  [bold blue]→ {d['name']}[/]({args}) {label(d['result'].get('source', '?'))} {_summary(d['result'])}{cached}")
    elif kind == "think" and d["text"].strip() and not d["calls"]:
        console.print(f"  [dim]agent: {d['text'][:300]}[/]")
    elif kind == "check":
        state = "[red]replanning[/]" if d["replan"] else ("[red]issues remain[/]" if d["errors"] else "[green]passed[/]")
        console.print(f"  [bold]✓ validator[/] {state}: {len(d['errors'])} errors, {len(d['warnings'])} warnings; "
                      f"covered {money(d['covered'])} of {money(d['budget'])}")
        for e in d["errors"]:
            console.print(f"    [red]✗ {e}[/]")
    elif kind == "retry":
        console.print(f"  [yellow]LLM busy ({d['error'][:80]}); retrying in {d['wait_s']}s[/]")
    elif kind == "repair":
        console.print(f"  [yellow]{d['step']}: malformed model output, asking it to repair[/]")
    elif kind == "patch":
        console.print(f"  [bold]change:[/] {d['note']} → re-search {d['affects']}")
    elif kind == "limit":
        console.print(f"  [yellow]{d['what']}[/]")
    elif kind in ("parsed", "composed"):
        console.print(f"  [dim]{kind}[/]")


def file_sink(path: Path):
    path.parent.mkdir(exist_ok=True)

    def sink(kind: str, d: dict) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": kind, **d}, default=str) + "\n")
    return sink


# --- Rendering ----------------------------------------------------------------------

def _results(state, tool: str) -> list[dict]:
    return [r for r in (state.get("results") or {}).values() if r["tool"] == tool]


def show_flights(state) -> None:
    for r in _results(state, "search_flights"):
        res, a = r["result"], r["args"]
        t = Table(title=f"Flights {a.get('origin')} → {a.get('destination')} on {a.get('date')} {label(res.get('source', '?'))}")
        if res.get("status") != "ok":
            console.print(f"{t.title}: {_summary(res)}")
            continue
        for c in ("ID", "Airline", "Depart", "Arrive", "Duration", "Stops", "Per person", "Δ trip vs ★"):
            t.add_column(c)
        chosen = {state["itinerary"].outbound_flight_id, state["itinerary"].return_flight_id} if state.get("itinerary") else set()
        base = next((f["price_per_person"] for f in res["items"] if f["id"] in chosen), None)
        pax = state["req"].travellers
        for f in res["items"]:
            t.add_row(("★ " if f["id"] in chosen else "") + f["id"], f["airline"], f["depart"], f["arrive"],
                      f"{f['duration_min'] // 60}h{f['duration_min'] % 60:02d}", str(f["stops"]), money(f["price_per_person"]),
                      delta((f["price_per_person"] - base) * pax) if base is not None else "")
        console.print(t)


def show_hotels(state) -> None:
    chosen = state["itinerary"].hotel_id if state.get("itinerary") else None
    for r in _results(state, "search_hotels"):
        res = r["result"]
        if res.get("status") != "ok":
            console.print(f"Hotels ({r['args']}): {_summary(res)}")
            continue
        filters = ", ".join(f"{k}={v}" for k, v in r["args"].items())
        t = Table(title=f"Hotels ({filters}) {label(res['source'])}")
        for c in ("ID", "Name", "Area", "Tier", "Per night", "Rooms", "Rating", "Amenities", "Δ stay vs ★"):
            t.add_column(c)
        nights = state["req"].nights
        ch = lookup(chosen, "H-") if chosen else None
        base = ch["price_per_night"] * math.ceil(state["req"].travellers / ch["capacity"]) * nights if ch else None
        for h in res["items"]:
            t.add_row(("★ " if h["id"] == chosen else "") + h["id"], h["name"], h["area"], h["tier"],
                      money(h["price_per_night"]), str(h["rooms_needed"]), str(h["rating"]), ", ".join(h["amenities"]),
                      delta(h["price_per_night"] * h["rooms_needed"] * nights - base) if base is not None else "")
        console.print(t)


def show_costs(state) -> None:
    c = state.get("costs")
    if not c:
        return
    t = Table(title="Cost breakdown (computed, not model-generated)")
    for col in ("Category", "Item", "Amount", "Source"):
        t.add_column(col)
    for l in c.lines:
        t.add_row(l.category, l.item, money(l.amount), label(l.source))
    console.print(t)
    width = 30
    for cat, amt in sorted(c.by_category.items(), key=lambda kv: -kv[1]):
        bar = "█" * max(1, round(width * amt / max(c.total, 1)))
        console.print(f"  {cat:<16}{money(amt):>10}  [blue]{bar}[/]")
    failed = {(r["tool"], r["result"].get("message")) for r in (state.get("results") or {}).values()
              if r["result"].get("status") == "error"}  # retries of one outage print once
    for tool, msg in sorted(failed):
        priced = tool in ("search_flights", "search_hotels")
        console.print(f"  {label('UNAVAILABLE')} {tool}: {msg}" + (" — not included in the total" if priced else ""))
    itin = state.get("itinerary")
    if itin and not (itin.outbound_flight_id and itin.return_flight_id):
        console.print(f"  {label('UNAVAILABLE')} round-trip flights not priced — total excludes intercity travel")
    if itin and not itin.hotel_id:
        console.print(f"  {label('UNAVAILABLE')} no hotel priced — total excludes accommodation")
    scope = "transport + accommodation" if c.budget_covers == "transport_and_accommodation" else "whole trip"
    incomplete = bool(itin and not (itin.outbound_flight_id and itin.return_flight_id and itin.hotel_id))
    ok = ("[red]OVER budget[/]" if not c.within_budget
          else "[yellow]can't confirm: total excludes unavailable items[/]" if incomplete
          else "[green]within budget[/]")
    console.print(f"  [bold]Total {money(c.total)}[/] · budget {money(c.budget)} covers {scope}: "
                  f"{money(c.covered_total)} → {ok}")


def show_plan(state) -> None:
    itin, req = state.get("itinerary"), state["req"]
    if not itin:
        return
    for i, day in enumerate(itin.days):
        t = Table(show_header=True, expand=True, title=f"Day {day.day} · {day.date} · {day.theme}")
        for col in ("Time", "Activity", "Why", "Travel (ESTIMATE)", "Cost (MOCK)"):
            t.add_column(col)
        stops = schedule(itin, i)
        if not stops:
            console.print(f"[bold]Day {day.day} · {day.date} · {day.theme}[/] — no activities")
        for s in stops:
            a = s["a"]
            t.add_row(f"{s['act'].start}–{s['end'] // 60:02d}:{s['end'] % 60:02d}",
                      f"[bold]{a['name']}[/] ({a['area']})\n[dim]{a['description']}[/]",
                      s["act"].rationale, f"{s['leg_min']} min / {money(s['leg_cost'])}" if s["leg_min"] else "-",
                      money(a["cost_per_person"] * req.travellers) if a["cost_per_person"] else "free")
        if stops:
            console.print(t)
        meals = []
        for m in day.meals:
            r = lookup(m.restaurant_id, "R-") if m.restaurant_id else None
            meals.append(f"{m.meal}: {r['name'] if r else 'hotel/casual'}{' — ' + m.note if m.note else ''}")
        console.print("  🍽  " + " · ".join(meals))
        console.print(f"  💤 {day.rest}")
        for tip in day.tips:
            console.print(f"  • {tip}")
    for w in _results(state, "get_weather"):
        res = w["result"]
        if res.get("status") == "ok" and res["source"] == "CLIMATE":
            console.print(f"Weather {label('CLIMATE')}: avg {res['avg_min_c']}–{res['avg_max_c']}°C, "
                          f"{res['avg_rain_mm_per_day']} mm rain/day, rainy days {res['rainy_day_share']:.0%}")
        elif res.get("status") == "ok":
            console.print(f"Weather {label('LIVE')} forecast: " + "; ".join(
                f"{d['date']} {d['min_c']}–{d['max_c']}°C {d['rain_mm']}mm" for d in res["days"]))
        else:
            console.print(f"Weather {label('UNAVAILABLE')}: {res.get('message')}")


def show_all(state) -> None:
    req = state["req"]
    console.print(Panel(f"{req.origin} → {req.destination} · {req.start_date:%d %b %Y} – {req.end_date:%d %b %Y} "
                        f"({req.days} days) · {req.adults} adults, {req.children} children · budget {money(req.budget)}\n"
                        + "\n".join(f"[yellow]assumption:[/] {a}" for a in req.assumptions), title="Trip"))
    if state.get("itinerary"):
        console.print(Panel(state["itinerary"].summary, title="Summary"))
    show_flights(state)
    show_hotels(state)
    show_plan(state)
    show_costs(state)
    for i in state.get("issues") or []:
        console.print(f"[{'red' if i.severity == 'error' else 'yellow'}]{i.severity}: {i.message}[/]")
    for u in state["itinerary"].unmet if state.get("itinerary") else []:
        console.print(f"[red]cannot satisfy:[/] {u}")
    console.print(LEGEND)


def show_diff(old, new) -> None:
    """What a refinement changed, so the user doesn't have to compare two full plans."""
    rows = []
    o, n = old["req"], new["req"]
    for field in ("start_date", "end_date", "adults", "children", "budget", "hotel_tier", "interests", "dietary", "accessibility"):
        if getattr(o, field) != getattr(n, field):
            rows.append((field, str(getattr(o, field)), str(getattr(n, field))))
    oi, ni = old.get("itinerary"), new.get("itinerary")
    def name(i):
        x = lookup(i) if i else None
        if not x:
            return str(i)
        return x.get("name") or f"{x['airline']} {x['depart']} ₹{x['price_per_person']:,}/pp"  # flights have no name
    if oi and ni:
        for field in ("outbound_flight_id", "return_flight_id", "hotel_id"):
            if getattr(oi, field) != getattr(ni, field):
                rows.append((field.replace("_id", ""), name(getattr(oi, field)), name(getattr(ni, field))))
        for k in range(max(len(oi.days), len(ni.days))):
            before = [name(a.attraction_id) for a in oi.days[k].activities] if k < len(oi.days) else []
            after = [name(a.attraction_id) for a in ni.days[k].activities] if k < len(ni.days) else []
            if before != after:
                rows.append((f"day {k + 1}", ", ".join(before) or "-", ", ".join(after) or "-"))
    oc, nc = old.get("costs"), new.get("costs")
    if oc and nc:
        for cat in sorted(set(oc.by_category) | set(nc.by_category)):
            a, b = oc.by_category.get(cat, 0), nc.by_category.get(cat, 0)
            if round(a) != round(b):
                rows.append((cat, money(a), f"{money(b)} ({delta(b - a)})"))
        rows.append(("total", money(oc.total), f"{money(nc.total)} ({delta(nc.total - oc.total)})"))
    t = Table(title=f"What changed: {new.get('change_note', '')}")
    for c in ("", "Before", "After"):
        t.add_column(c)
    for r in rows:
        t.add_row(*r)
    console.print(t)


def as_json(state) -> str:
    return json.dumps({"requirements": state["req"].model_dump(mode="json"),
                       "itinerary": state["itinerary"].model_dump(mode="json") if state.get("itinerary") else None,
                       "costs": state["costs"].model_dump(mode="json") if state.get("costs") else None,
                       "issues": [i.model_dump() for i in state.get("issues") or []]}, indent=2, ensure_ascii=False)


# --- Loop ---------------------------------------------------------------------------

def plan(state: dict) -> dict | None:
    try:
        with console.status("[bold]agent working…[/]"):
            return graph.run(state)
    except (graph.LLMUnavailable, graph.MalformedOutput) as e:
        console.print(f"[red]Model problem:[/] {e}\nNothing was changed; please try again in a minute.")
        return None


def book(state, trip_id: str) -> None:
    """Explicit confirmation gate. The agent has no booking tool; only the user can trigger this."""
    itin = state.get("itinerary")
    if not itin or any(i.severity == "error" for i in state.get("issues") or []):
        console.print("[red]Nothing bookable: the plan has unresolved errors.[/]")
        return
    show_costs(state)
    console.print("[bold]This is a MOCK booking: no real reservation or payment will be made.[/]")
    if console.input("Type [bold]yes[/] to confirm the mock booking: ").strip().lower() != "yes":
        console.print("Cancelled. Nothing was booked.")
        return
    result = mock_book(trip_id, [itin.outbound_flight_id, itin.return_flight_id, itin.hotel_id])
    if result.get("status") == "confirmed":
        console.print(f"[magenta]MOCK[/] booking reference {result['reference']} — no real booking was made.")
    else:
        console.print(f"[red]Booking not confirmed:[/] {result.get('message')}. Nothing was booked.")


def main() -> None:
    as_json_only = "--json" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--json"]
    if args[:1] == ["--trips"]:
        for tid, title, updated in store.list_trips():
            console.print(f"{tid}  {title}  [dim](updated {time.strftime('%Y-%m-%d %H:%M', time.localtime(updated))})[/]")
        return
    resume = args[1] if args[:1] == ["--resume"] and len(args) > 1 else None
    trip_id = resume or uuid.uuid4().hex[:8]
    graph.SINKS[:] = [console_sink, file_sink(Path("runs") / f"{trip_id}.jsonl")]
    console.print(f"[bold]Agentic Travel Planner[/] · trip {trip_id} · trace: runs/{trip_id}.jsonl\n{LEGEND}")

    if resume:
        loaded = store.load(resume)
        if not loaded:
            console.print(f"[red]No saved trip {resume}.[/] Use --trips to list them.")
            return
        state, history = loaded
    else:
        text = " ".join(args) or console.input("[bold]Describe your trip:[/] ")
        while True:  # clarification loop until we have a plan
            new = plan({"mode": "new", "text": text})
            if new is None:
                return
            if new.get("question"):
                answer = console.input(f"[bold yellow]?[/] {new['question']}\n> ")
                text = f"{text}\nAdditional details: {answer}"
                continue
            state, history = new, [{"request": text, "at": time.time()}]
            store.save(trip_id, state, history)
            break

    if as_json_only:
        print(as_json(state))
        return
    show_all(state)
    console.print(HELP)
    while True:
        cmd = console.input("\n[bold]> [/]").strip()
        if cmd in ("quit", "exit", "q"):
            break
        if cmd == "":
            continue
        if cmd in ("flights", "hotels", "costs", "plan", "json"):
            {"flights": show_flights, "hotels": show_hotels, "costs": show_costs, "plan": show_plan,
             "json": lambda s: print(as_json(s))}[cmd](state)
            continue
        if cmd == "book":
            book(state, trip_id)
            continue
        new = plan({**state, "mode": "refine", "text": cmd})
        if new is None:
            continue
        if new.get("question"):
            console.print(f"[yellow]?[/] {new['question']} (plan unchanged; rephrase your change)")
            continue
        old, state = state, new
        history.append({"change": cmd, "note": state.get("change_note"), "at": time.time()})
        store.save(trip_id, state, history)  # saved only after a successful revision
        show_all(state)
        show_diff(old, state)
    console.print(f"Saved. Resume with: python main.py --resume {trip_id}")


if __name__ == "__main__":
    main()
