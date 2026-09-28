"""Frozen GPT-5.6-Luna function-call controller for legal high-level actions."""

from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen


class LunaToolController:
    def __init__(self, *, requester=None) -> None:
        self.requester = requester or self._request

    @staticmethod
    def _request(payload: dict) -> dict:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required for the Luna controller")
        request = Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=60) as response:
            return json.load(response)

    def choose(self, legal_actions: list[str], public_context: dict) -> str:
        if not legal_actions:
            raise ValueError("controller needs at least one legal action")
        forbidden = ("private", "ground_truth", "true_state", "target_position_xyz",
                     "oracle_shortest_path")
        if any(token in str(key).lower() for key in public_context for token in forbidden):
            raise ValueError("evaluator-private field in controller context")
        payload = {
            "model": "gpt-5.6-luna",
            "input": [
                {"role": "system", "content": (
                    "You are a frozen embodied-search tool controller. Choose one legal high-level "
                    "action using only the supplied public belief, memory, costs and evidence. "
                    "Prefer the maximum paper utility and never infer evaluator-private truth."
                )},
                {"role": "user", "content": json.dumps(public_context, ensure_ascii=False)},
            ],
            "tools": [{
                "type": "function", "name": "select_action",
                "description": "Choose one legal EvolvingNav high-level action.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"action": {"type": "string", "enum": legal_actions}},
                    "required": ["action"], "additionalProperties": False,
                },
            }],
            "tool_choice": {"type": "function", "name": "select_action"},
        }
        response = self.requester(payload)
        calls = [item for item in response.get("output", [])
                 if item.get("type") == "function_call" and item.get("name") == "select_action"]
        if len(calls) != 1:
            raise ValueError("Luna did not return exactly one select_action call")
        action = json.loads(calls[0]["arguments"])["action"]
        if action not in legal_actions:
            raise ValueError("Luna selected an illegal action")
        return action
