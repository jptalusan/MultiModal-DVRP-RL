"""Baseline policy tests. Offline — synthetic States, no env/network."""

from __future__ import annotations

from dvrp_core.models.core import NodeLocation, Request, State, Trip, Vehicle

from dvrp_rl.policies import AcceptAll, RandomPolicy


def _state() -> State:
    req = Request(
        id=0, origin=NodeLocation(0), destination=NodeLocation(1), passengers=2,
        received_time=100.0, earliest_pickup=100.0, latest_dropoff=460.0,
    )
    veh = Vehicle(id=0, depot_id=0, location=NodeLocation(0), capacity=4)
    return State(
        current_time=100.0, vehicles=[veh], accepted_trips={}, passengers={},
        new_request=req, transit_routes=[], bus_states=[],
    )


def test_accept_all_returns_a_trip_built_from_the_request():
    state = _state()
    action = AcceptAll().create_trips(state)
    assert isinstance(action, Trip)
    r = state.new_request
    assert (action.origin, action.destination, action.passengers) == (r.origin, r.destination, r.passengers)
    assert (action.earliest_pickup, action.latest_dropoff) == (r.earliest_pickup, r.latest_dropoff)


def test_random_p0_always_rejects_and_p1_always_accepts():
    state = _state()
    assert all(RandomPolicy(accept_prob=0.0, seed=s).create_trips(state) is None for s in range(5))
    assert all(isinstance(RandomPolicy(accept_prob=1.0, seed=s).create_trips(state), Trip) for s in range(5))


def test_random_is_reproducible_for_a_given_seed():
    state = _state()

    def seq():
        return [RandomPolicy(accept_prob=0.5, seed=42).accept(state) for _ in range(20)]

    assert seq() == seq()


def test_trip_ids_are_unique_across_accepts():
    state = _state()
    policy = AcceptAll()
    ids = {policy.create_trips(state).id for _ in range(10)}
    assert len(ids) == 10
