from typing import List, Dict, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation

class WasteAction(Action):
    """What the agent does each step."""
    action_type: str = Field(..., description="Type of action: 'move' or 'collect'")
    target_node: int = Field(..., description="Node ID to move to or collect from")
    message: Optional[str] = Field(default=None, description="Optional text message for LLM agents")

class BinInfo(object):
    """Info about a single bin."""
    def __init__(self, node_id: int, fill_level: float, capacity: float = 1.0):
        self.node_id = node_id
        self.fill_level = fill_level  # 0.0 to 1.0
        self.capacity = capacity

class WasteObservation(Observation):
    """What the agent sees each step."""
    current_node: int = Field(..., description="Current truck location (node ID)")
    bin_levels: Dict[int, float] = Field(..., description="Fill level of each bin (0.0-1.0)")
    fuel_remaining: float = Field(..., description="Remaining fuel (0.0-1.0)")
    collected_bins: List[int] = Field(default=[], description="List of bin IDs collected so far")
    step_count: int = Field(default=0, description="Current step number")
    total_reward: float = Field(default=0.0, description="Accumulated reward so far")
    done: bool = Field(default=False, description="Is episode over?")
    message: str = Field(default="", description="Human readable description of what happened")
    graph_edges: List[List[int]] = Field(default=[], description="List of [from, to, distance] edges")
