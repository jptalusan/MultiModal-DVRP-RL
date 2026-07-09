"""Base class for on-demand accept/reject policies.

For ``on_demand_only``, a policy's decision is binary: accept the new
request (return a ``Trip`` — the greedy solver then inserts it) or reject
it (return ``None``). This base factors the Trip-building once; subclasses
only implement ``accept(state) -> bool``. Our baselines (AcceptAll,
Random) and later learned policies (MCTS, RL) all subclass it.
"""

from __future__ import annotations

from abc import abstractmethod

from dvrp_core.models.core import Action, State, Trip
from dvrp_core.policy import Policy


class AcceptRejectPolicy(Policy):
    """A ``Policy`` whose only choice is accept vs reject the new request."""

    @abstractmethod
    def accept(self, state: State) -> bool:
        """Return True to serve ``state.new_request``, False to reject it."""
        ...

    def create_trips(self, state: State) -> Action:
        if not self.accept(state):
            return None
        r = state.new_request
        return Trip(
            id=self.generate_trip_id(),
            origin=r.origin,
            destination=r.destination,
            passengers=r.passengers,
            earliest_pickup=r.earliest_pickup,
            latest_dropoff=r.latest_dropoff,
        )
