"""Brief §7 scenarios through the real graph with a scripted model: proves orchestration
(routing, tool execution, caching, validation loop, refinement) without network or quota."""
from datetime import date

import pytest
from langchain_core.messages import AIMessage

from travel import graph
from travel.models import ParsedRequest, RequirementPatch

from test_logic import GOA, _plan

REQUEST = "Plan a 5-day trip from Indore to Goa ..."  # content is irrelevant to the scripted model


class FixedDate(date):
    @classmethod
    def today(cls):
        return date(2026, 10, 30)


class Scripted:
    """Stands in for the chat model. Each call pops the next scripted answer for its kind:
    'tools' (research turns) or a schema name (parse/compose/refine)."""

    def __init__(self, script: dict[str, list]):
        self.script, self.calls = script, {k: 0 for k in script}

    def with_structured_output(self, schema, include_raw=True):
        return _Runner(self, schema.__name__)

    def bind_tools(self, tools):
        return _Runner(self, "tools")


class _Runner:
    def __init__(self, llm, kind):
        self.llm, self.kind = llm, kind

    def invoke(self, messages):
        self.llm.calls[self.kind] += 1
        out = self.llm.script[self.kind].pop(0)
        if self.kind == "tools":
            return out
        if isinstance(out, str):  # simulate malformed structured output
            return {"raw": None, "parsed": None, "parsing_error": out}
        return {"raw": None, "parsed": out, "parsing_error": None}


def calls(*specs):
    return AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": f"c{i}"} for i, (n, a) in enumerate(specs)])


FLIGHTS = [("search_flights", {"origin": "Indore", "destination": "Goa", "date": "2026-12-15", "passengers": 2}),
           ("search_flights", {"origin": "Goa", "destination": "Indore", "date": "2026-12-19", "passengers": 2})]
RESEARCH = [*FLIGHTS, ("search_hotels", {"city": "Goa", "guests": 2, "tier": "mid-range"}),
            ("search_attractions", {"city": "Goa", "interests": GOA["interests"]}),
            ("search_restaurants", {"city": "Goa"})]


@pytest.fixture
def run(monkeypatch):
    events = []
    monkeypatch.setattr(graph, "date", FixedDate)
    monkeypatch.setattr(graph, "SINKS", [lambda k, d: events.append((k, d))])

    def go(script, state=None, text=REQUEST, mode="new"):
        llm = Scripted(script)
        monkeypatch.setattr(graph, "get_llm", lambda name: llm)
        events.clear()
        return graph.run({**(state or {}), "mode": mode, "text": text}), llm, events
    return go


def plan_script(**over):
    return {"ParsedRequest": [ParsedRequest(**GOA | over)],
            "tools": [calls(*RESEARCH), AIMessage(content="Recommend FL-...-0, H-GOA-04.")],
            "Itinerary": [_plan()]}


def test_T1_normal_request(run):
    state, llm, events = run(plan_script())
    tools = [d["name"] for k, d in events if k == "tool"]
    assert tools == [n for n, _ in RESEARCH]
    assert not [i for i in state["issues"] if i.severity == "error"]
    assert state["costs"].within_budget and state["costs"].covered_total <= 35000
    assert llm.calls == {"ParsedRequest": 1, "tools": 2, "Itinerary": 1}


def test_T2_missing_information_asks_instead_of_planning(run):
    state, llm, _ = run({"ParsedRequest": [ParsedRequest(destination="Goa", interests=["beaches"])],
                         "tools": [], "Itinerary": []})
    assert "travelling from" in state["question"] and "budget" in state["question"]
    assert llm.calls["tools"] == 0 and not state.get("itinerary")


def test_T3_unsatisfiable_budget_replans_then_reports_honestly(run):
    script = plan_script(budget=5000)
    script["tools"] += [AIMessage(content="Nothing cheaper exists."), AIMessage(content="Still nothing cheaper.")]
    script["Itinerary"] = [_plan(), _plan(), _plan()]
    state, llm, events = run(script)
    checks = [d for k, d in events if k == "check"]
    assert [c["replan"] for c in checks] == [True, True, False]       # bounded: MAX_REPLANS
    assert llm.calls["Itinerary"] == 3
    assert any(i.code == "budget" for i in state["issues"])           # reported, not hidden


def test_T4_failed_tool_degrades_gracefully(run, monkeypatch):
    script = plan_script()
    script["Itinerary"] = [_plan(outbound_flight_id=None, return_flight_id=None)]
    monkeypatch.setenv("MOCK_FAIL", "search_flights")
    state, _, events = run(script)
    flight_results = [d["result"] for k, d in events if k == "tool" and d["name"] == "search_flights"]
    assert all(r["status"] == "error" and r["source"] == "UNAVAILABLE" for r in flight_results)
    assert any(i.code == "no_flights" for i in state["issues"])
    assert not any(l.category == "transport" and "Flight" in l.item for l in state["costs"].lines)


def test_T5_refinement_reuses_unaffected_results(run):
    first, _, _ = run(plan_script())
    cheaper = _plan(hotel_id="H-GOA-05")
    state, llm, events = run({
        "RequirementPatch": [RequirementPatch(affects=["hotel"], note="cheaper hotel")],
        "tools": [calls(*FLIGHTS, ("search_hotels", {"city": "Goa", "guests": 2, "max_price_per_night": 2700})),
                  AIMessage(content="H-GOA-05 is cheaper.")],
        "Itinerary": [cheaper]}, state=first, text="find a cheaper hotel", mode="refine")
    cached = {(d["name"], d["cached"]) for k, d in events if k == "tool"}
    assert cached == {("search_flights", True), ("search_hotels", False)}  # flights not re-fetched
    assert state["req"] == first["req"]                                    # other preferences preserved
    assert state["itinerary"].hotel_id == "H-GOA-05" and state["previous"].hotel_id == "H-GOA-04"


def test_refinement_with_invalid_change_keeps_plan(run):
    first, _, _ = run(plan_script())
    state, llm, _ = run({"RequirementPatch": [RequirementPatch(end_date="2026-12-10", affects=["flights"], note="x")],
                         "tools": [], "Itinerary": []}, state=first, text="end on 10 Dec", mode="refine")
    assert "before the start date" in state["question"] and llm.calls["tools"] == 0


def test_malformed_model_output_is_repaired_once(run):
    script = plan_script()
    script["Itinerary"] = ["days: field required", _plan()]
    state, llm, events = run(script)
    assert [k for k, _ in events].count("repair") == 1 and state["itinerary"]


def test_malformed_twice_fails_cleanly(run):
    script = plan_script()
    script["Itinerary"] = ["bad", "still bad"]
    with pytest.raises(graph.MalformedOutput):
        run(script)


def test_hallucinated_id_is_rejected(run):
    script = plan_script()
    bad = _plan()
    bad.days[1].activities[0].attraction_id = "A-GOA-99"
    script["tools"] += [AIMessage(content="ok")] * 2
    script["Itinerary"] = [bad, _plan(), _plan()]
    state, _, events = run(script)
    first_check = next(d for k, d in events if k == "check")
    assert any("A-GOA-99" in e for e in first_check["errors"]) and first_check["replan"]
    assert not [i for i in state["issues"] if i.severity == "error"]  # fixed on the replan


def test_injected_instructions_stay_inside_tool_data(run):
    state, _, _ = run(plan_script())
    tool_msgs = [m for m in state["messages"] if m.type == "tool"]
    injected = next(m.content for m in tool_msgs if "ignore all previous" in m.content)
    assert injected.startswith('<tool_data tool="search_attractions"') and injected.endswith("</tool_data>")
    assert "untrusted data" in state["messages"][0].content  # system prompt tells the model so


def test_research_loop_is_bounded(run):
    script = plan_script()
    script["tools"] = [calls(*RESEARCH)] * (graph.MAX_TOOL_ROUNDS + 1)  # model never stops calling tools
    state, llm, events = run(script)
    assert llm.calls["tools"] == graph.MAX_TOOL_ROUNDS
    assert any(k == "limit" for k, _ in events)


def test_saved_trip_round_trips_and_can_be_refined(run, tmp_path, monkeypatch):
    from travel import store
    monkeypatch.setattr(store, "DB_PATH", str(tmp_path / "t.db"))
    first, _, _ = run(plan_script())
    store.save("abc", first, [{"request": REQUEST}])
    state, history = store.load("abc")
    assert state["req"] == first["req"] and state["itinerary"] == first["itinerary"] and history[0]["request"] == REQUEST
    assert store.list_trips()[0][0] == "abc"
    refined, _, events = run({"RequirementPatch": [RequirementPatch(affects=["hotel"], note="cheaper")],
                              "tools": [calls(FLIGHTS[0]), AIMessage(content="ok")],
                              "Itinerary": [_plan(hotel_id="H-GOA-05")]}, state=state, text="cheaper", mode="refine")
    assert any(d["cached"] for k, d in events if k == "tool")       # cache survived the reload
    assert refined["itinerary"].hotel_id == "H-GOA-05"


def test_booking_is_mock_and_never_an_agent_tool():
    from travel.tools import TOOLS, mock_book
    assert "mock_book" not in {t.name for t in TOOLS}
    assert mock_book("abc", ["H-GOA-04"])["reference"] == "MOCK-ABC"
    assert mock_book("abc", ["H-GOA-04", None])["status"] == "error"


def test_diff_names_changed_flights(run):
    # Seen live: "find a cheaper flight" crashed show_diff with KeyError 'name' (flights have no name).
    import main
    first, _, _ = run(plan_script())
    new = dict(first, itinerary=_plan(outbound_flight_id="FL-IDR-GOI-20261215-1", hotel_id="H-GOA-05"))
    main.show_diff(first, new)
