import heapq
import random
from uuid import uuid4
from typing import Optional, Dict, List, Any
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import WasteAction, WasteObservation
except ImportError:
    from models import WasteAction, WasteObservation


def dijkstra(graph: Dict, start: int) -> Dict[int, float]:
    """Shortest distance from start to all nodes."""
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    heap = [(0, start)]
    while heap:
        dist, node = heapq.heappop(heap)
        if dist > distances[node]:
            continue
        for neighbour, weight in graph[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbour]:
                distances[neighbour] = new_dist
                heapq.heappush(heap, (new_dist, neighbour))
    return distances


def build_graph(edges: List) -> Dict:
    """Build adjacency list from edge list."""
    graph = {}
    for u, v, w in edges:
        if u not in graph: graph[u] = []
        if v not in graph: graph[v] = []
        graph[u].append((v, w))
        graph[v].append((u, w))
    return graph


class WasteRouteEnvironment(Environment):
    """
    WasteRoute-Env: Urban Waste Collection Environment.
    A truck navigates a city graph to collect waste bins.
    Agent must prioritize full bins while managing fuel efficiently.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    TASKS = {
        "easy": {
            "description": "5 bins, full fuel. Collect all high-fill bins.",
            "fuel": 1.0,
            "dynamic": False,
            "max_steps": 30,
            "edges": [
                (0,1,2),(0,2,3),(1,3,2),
                (2,3,1),(3,4,2),(2,5,3),
            ],
            "bin_nodes": [1, 2, 3, 4, 5],
        },
        "medium": {
            "description": "15 bins, limited fuel. Prioritize full bins wisely.",
            "fuel": 0.7,
            "dynamic": False,
            "max_steps": 60,
            "edges": [
                (0,1,2),(0,2,3),(1,3,2),(2,3,1),(3,4,2),
                (2,5,3),(4,6,2),(5,6,1),(6,7,3),(7,8,2),
                (8,9,1),(9,10,2),(10,11,3),(11,12,2),(12,13,1),
                (13,15,2),(14,15,3),(0,14,4),(7,14,2),(10,15,3),
            ],
            "bin_nodes": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        },
        "hard": {
            "description": "20 bins, low fuel, dynamic fills, overflow penalties.",
            "fuel": 0.5,
            "dynamic": True,
            "max_steps": 80,
            "edges": [
                (0,1,2),(0,2,3),(1,3,2),(2,3,1),(3,4,2),
                (2,5,3),(4,6,2),(5,6,1),(6,7,3),(7,8,2),
                (8,9,1),(9,10,2),(10,11,3),(11,12,2),(12,13,1),
                (13,15,2),(14,15,3),(0,14,4),(7,14,2),(10,15,3),
                (15,16,2),(16,17,1),(17,18,3),(18,19,2),(19,20,1),
                (16,20,4),(0,20,5),(9,16,3),(12,19,2),(5,17,4),
            ],
            "bin_nodes": list(range(1, 21)),
        },
    }

    def __init__(self, task: str = "easy"):
        self._task = task if task in self.TASKS else "easy"
        self._config = self.TASKS[self._task]
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._graph = {}
        self._bin_levels = {}
        self._collected = []
        self._fuel = 1.0
        self._current_node = 0
        self._total_reward = 0.0

    def _randomize_bin_levels(self) -> Dict[int, float]:
        """
        Randomize bin fill levels every episode.
        Ensures agent must READ and DECIDE, not memorize.
        """
        return {
            node: round(random.uniform(0.2, 1.0), 2)
            for node in self._config["bin_nodes"]
        }

    # def reset(self, task: Optional[str] = None, **kwargs) -> WasteObservation:
    #     """Reset environment for a new episode."""
    #     if task and task in self.TASKS:
    #         self._task = task
    #         self._config = self.TASKS[task]

    #     self._state = State(episode_id=str(uuid4()), step_count=0)
    #     self._graph = build_graph(self._config["edges"])

    #     # 🎲 Randomize bin levels every episode!
    #     self._bin_levels = self._randomize_bin_levels()

    #     self._collected = []
    #     self._fuel = self._config["fuel"]
    #     self._current_node = 0
    #     self._total_reward = 0.0


    def reset(self, task: Optional[str] = None, seed: Optional[int] = None, **kwargs) -> WasteObservation:
        """Reset environment for a new episode."""
        
        # Lock the randomizer ONLY if a seed is provided
        if seed is not None:
            random.seed(seed)

        if task and task in self.TASKS:
            self._task = task
            self._config = self.TASKS[task]

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._graph = build_graph(self._config["edges"])

        # Randomize bin levels (will be perfectly deterministic if seed was set above)
        self._bin_levels = self._randomize_bin_levels()

        self._collected = []
        self._fuel = self._config["fuel"]
        self._current_node = 0
        self._total_reward = 0.0


    # def reset(self, task: Optional[str] = None, seed: int = 42, **kwargs):
    #     if task and task in self.TASKS:
    #         self._task = task
    #         self._config = self.TASKS[task]

    # # Seed FIRST for reproducibility
    #     random.seed(seed)

    #     self._state = State(episode_id=str(uuid4()), step_count=0)
    #     self._graph = build_graph(self._config["edges"])
    
    # # Now randomize — but deterministically because of seed above
    #     self._bin_levels = self._randomize_bin_levels()
    
    #     self._collected = []
    #     self._fuel = self._config["fuel"]
    #     self._current_node = 0
    #     self._total_reward = 0.0

        return WasteObservation(
            current_node=self._current_node,
            bin_levels=dict(self._bin_levels),
            fuel_remaining=round(self._fuel, 3),
            collected_bins=list(self._collected),
            step_count=0,
            total_reward=0.0,
            done=False,
            message=f"Episode started. Task: {self._task}. Truck at depot (node 0).",
            graph_edges=self._config["edges"],
        )
    


    def step(self, action: WasteAction, **kwargs) -> WasteObservation:
        """Execute one step."""
        self._state.step_count += 1
        reward = 0.0
        message = ""

        # Parse text message from LLM agents
        if action.message:
            parsed = self._parse_text_action(action.message)
            if parsed:
                action = parsed

        # Get distances from current node
        distances = dijkstra(self._graph, self._current_node)
        target = action.target_node

        if target not in distances or distances[target] == float('inf'):
            message = f"Node {target} unreachable from {self._current_node}!"
            reward = -0.5
        else:
            # Fuel cost = travel distance normalized
            fuel_used = distances[target] / 20.0
            self._fuel = max(0.0, self._fuel - fuel_used)
            self._current_node = target

            if action.action_type == "collect":
                if target in self._bin_levels and target not in self._collected:
                    fill = self._bin_levels[target]
                    if fill >= 0.7:
                        reward = fill          # high reward for full bins
                        message = f"Collected bin {target} (fill={fill:.2f}) ✅"
                    else:
                        reward = fill * 0.3    # low reward for near-empty bins
                        message = f"Collected bin {target} (fill={fill:.2f}) — bin was low."
                    self._collected.append(target)
                    self._bin_levels[target] = 0.0
                elif target in self._collected:
                    reward = -0.3
                    message = f"Bin {target} already collected! Wasted trip."
                else:
                    reward = -0.1
                    message = f"No bin at node {target}!"
            else:
                reward = -0.05  # small penalty for just moving
                message = f"Moved to node {target}."

        # Dynamic bin levels for hard task
        if self._config.get("dynamic"):
            for bin_id in list(self._bin_levels.keys()):
                if self._bin_levels[bin_id] > 0:
                    self._bin_levels[bin_id] = min(
                        1.0, self._bin_levels[bin_id] + random.uniform(0.01, 0.05)
                    )
                    if self._bin_levels[bin_id] >= 1.0:
                        reward -= 0.5
                        message += f" ⚠️ Bin {bin_id} OVERFLOWED!"

        # Fuel empty penalty
        if self._fuel <= 0:
            reward -= 1.0
            message += " 🚨 OUT OF FUEL!"

        self._total_reward += reward

        # Episode done conditions
        high_fill_bins = [
            n for n in self._config["bin_nodes"]
            if self._config.get("bin_levels", {}).get(n, 1.0) >= 0.7
        ]
        all_collected = all(n in self._collected for n in high_fill_bins)
        done = (
            self._fuel <= 0
            or self._state.step_count >= self._config["max_steps"]
            or all_collected
        )

        return WasteObservation(
            current_node=self._current_node,
            bin_levels=dict(self._bin_levels),
            fuel_remaining=round(self._fuel, 3),
            collected_bins=list(self._collected),
            step_count=self._state.step_count,
            total_reward=round(self._total_reward, 3),
            done=done,
            message=message,
            graph_edges=self._config["edges"],
        )

    def _parse_text_action(self, message: str) -> Optional[WasteAction]:
        """Parse text commands from LLM agents."""
        message = message.lower().strip()
        if "collect" in message:
            for word in message.split():
                if word.isdigit():
                    return WasteAction(action_type="collect", target_node=int(word))
        elif "move" in message or "go" in message:
            for word in message.split():
                if word.isdigit():
                    return WasteAction(action_type="move", target_node=int(word))
        return None

    @property
    def state(self) -> State:
        return self._state
