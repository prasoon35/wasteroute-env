# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""WasteRoute Environment Client."""

from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import WasteAction, WasteObservation


class WasteRouteEnv(EnvClient[WasteAction, WasteObservation, State]):
    """
    Client for the WasteRoute Environment.

    Example:
        >>> with WasteRouteEnv(base_url="http://localhost:7860") as client:
        ...     result = client.reset()
        ...     print(result.observation.bin_levels)
        ...
        ...     result = client.step(WasteAction(action_type="move", target_node=1))
        ...     print(result.observation.current_node)
    """

    def _step_payload(self, action: WasteAction) -> Dict:
        """Convert WasteAction to JSON payload."""
        return {
            "action_type": action.action_type,
            "target_node": action.target_node,
            "message": action.message,
        }

    def _parse_result(self, payload: Dict) -> StepResult[WasteObservation]:
        """Parse server response into StepResult[WasteObservation]."""
        obs_data = payload.get("observation", {})
        observation = WasteObservation(
            current_node=obs_data.get("current_node", 0),
            bin_levels=obs_data.get("bin_levels", {}),
            fuel_remaining=obs_data.get("fuel_remaining", 1.0),
            collected_bins=obs_data.get("collected_bins", []),
            step_count=obs_data.get("step_count", 0),
            total_reward=obs_data.get("total_reward", 0.0),
            done=payload.get("done", False),
            message=obs_data.get("message", ""),
            graph_edges=obs_data.get("graph_edges", []),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
