"""
Mission 06 — Part 2: Communications & Protocol Officer
Tool Protocol, JSON-Schema Validator & Action Dispatcher with Error Boundary

Sector SEC-06-CAR (Eta Carinae & Homunculus Nebula)
Subsystem 06: Autonomous Warp State Machine & Deliberation Loop

Flow for every tool request from the LLM:
    raw JSON text -> parse -> validate against the tool's schema -> execute
Any failure at any stage becomes an observation string
    "Tool Execution Error: <msg>"
which is injected into the agent's context so the next deliberation cycle can
fix the mistake. Nothing ever propagates as a Python exception.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# 1. Tools
# ---------------------------------------------------------------------------
def navigate_warp_vector(x: float, y: float, z: float) -> str:
    """Set the warp course vector through the nebula."""
    return f"Warp vector aligned to [{x:.2f}, {y:.2f}, {z:.2f}]."


def adjust_plasma_shields(intensity: int) -> str:
    """Set plasma shield intensity in percent (0..100)."""
    if not 0 <= intensity <= 100:
        raise ValueError(f"Shield intensity {intensity}% out of range [0, 100].")
    return f"Plasma shields set to {intensity}%."


def query_sensor_array(target: str) -> str:
    """Scan a target in the nebula."""
    readings = {
        "homunculus_core": "Severe plasma flare. Dense stellar dust at azimuth 142.",
        "eta_carinae_b": "O-type companion emitting ~100x solar luminosity.",
        "escape_corridor": "Clear gravitational corridor along vector [-0.85, 0.52, 0.05].",
    }
    key = target.strip().lower()
    if not key:
        raise ValueError("Sensor target must not be empty.")
    return readings.get(key, f"Target '{target}' unresolved.")


# ---------------------------------------------------------------------------
# 2. Tool contracts (JSON-Schema subset) + a strict validator (no external libs)
# ---------------------------------------------------------------------------
TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "navigate_warp_vector": {
        "type": "object",
        "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}},
        "required": ["x", "y", "z"],
        "additionalProperties": False,
    },
    "adjust_plasma_shields": {
        "type": "object",
        # Range check is left to the tool on purpose, so the error boundary
        # can be demonstrated with the ValueError from the brief.
        "properties": {"intensity": {"type": "integer"}},
        "required": ["intensity"],
        "additionalProperties": False,
    },
    "query_sensor_array": {
        "type": "object",
        "properties": {"target": {"type": "string", "minLength": 1}},
        "required": ["target"],
        "additionalProperties": False,
    },
}

TOOL_REGISTRY: Dict[str, Callable[..., str]] = {
    "navigate_warp_vector": navigate_warp_vector,
    "adjust_plasma_shields": adjust_plasma_shields,
    "query_sensor_array": query_sensor_array,
}


def _type_ok(value: Any, expected: str) -> bool:
    # bool is a subclass of int in Python, so it is excluded explicitly.
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    return False


def validate_against_schema(args: Any, schema: Dict[str, Any]) -> List[str]:
    """Return a list of violations (empty list = valid)."""
    errors: List[str] = []
    if not _type_ok(args, schema.get("type", "object")):
        return [f"arguments must be a JSON {schema.get('type', 'object')}, got {type(args).__name__}"]

    props = schema.get("properties", {})
    for req in schema.get("required", []):
        if req not in args:
            errors.append(f"missing required argument '{req}'")
    if schema.get("additionalProperties", True) is False:
        for key in args:
            if key not in props:
                errors.append(f"unexpected argument '{key}'")
    for key, rule in props.items():
        if key not in args:
            continue
        val = args[key]
        if not _type_ok(val, rule["type"]):
            errors.append(f"'{key}' must be {rule['type']}, got {type(val).__name__} ({val!r})")
            continue
        if rule["type"] in ("number", "integer") and val != val:  # NaN
            errors.append(f"'{key}' must not be NaN")
        if "minimum" in rule and val < rule["minimum"]:
            errors.append(f"'{key}' must be >= {rule['minimum']}")
        if "maximum" in rule and val > rule["maximum"]:
            errors.append(f"'{key}' must be <= {rule['maximum']}")
        if "minLength" in rule and len(val) < rule["minLength"]:
            errors.append(f"'{key}' must have length >= {rule['minLength']}")
    return errors


# ---------------------------------------------------------------------------
# 3. Action dispatcher with error boundary
# ---------------------------------------------------------------------------
ERROR_PREFIX = "Tool Execution Error: "


class ActionDispatcher:
    """Parse -> validate -> execute. Always returns a string observation."""

    def __init__(self, registry=None, schemas=None):
        self.registry = registry if registry is not None else TOOL_REGISTRY
        self.schemas = schemas if schemas is not None else TOOL_SCHEMAS
        self.log: List[Dict[str, Any]] = []

    def _error(self, msg: str) -> str:
        return ERROR_PREFIX + msg

    def dispatch(self, request: Union[str, Dict[str, Any]], arguments: Optional[Any] = None) -> str:
        """
        Accepts either
          dispatch('{"tool": "...", "arguments": {...}}')        # raw LLM JSON
          dispatch({"tool": "...", "arguments": {...}})
          dispatch("tool_name", {...})
        """
        try:
            tool, args = self._parse(request, arguments)
            if isinstance(tool, str) and tool.startswith(ERROR_PREFIX):
                obs = tool
            elif tool not in self.registry:
                obs = self._error(f"unknown tool '{tool}'. Available: {sorted(self.registry)}")
            else:
                problems = validate_against_schema(args, self.schemas[tool])
                if problems:
                    obs = self._error(f"invalid arguments for '{tool}': " + "; ".join(problems))
                else:
                    obs = "Observation: " + self.registry[tool](**args)
        except Exception as exc:  # the error boundary: nothing escapes
            obs = self._error(f"{type(exc).__name__}: {exc}")
        self.log.append({"request": request if arguments is None else (request, arguments), "observation": obs})
        return obs

    def _parse(self, request, arguments):
        if arguments is not None:  # dispatch("tool", {...})
            return request, arguments
        if isinstance(request, str):
            try:
                request = json.loads(request)
            except json.JSONDecodeError as exc:
                return self._error(f"malformed JSON payload ({exc.msg} at pos {exc.pos})"), None
        if not isinstance(request, dict):
            return self._error("payload must be a JSON object with 'tool' and 'arguments'"), None
        if "tool" not in request:
            return self._error("payload missing 'tool' field"), None
        return request["tool"], request.get("arguments", {})

    def dispatch_into(self, agent, request, arguments=None) -> str:
        """Dispatch and inject the observation (or error) into agent.context."""
        obs = self.dispatch(request, arguments)
        agent.context.append({
            "role": "observation",
            "is_error": obs.startswith(ERROR_PREFIX),
            "content": obs,
        })
        return obs


if __name__ == "__main__":
    d = ActionDispatcher()
    print("=== Subsystem 06 / Part 2: Tool Protocol & Dispatcher ===\n")
    cases = [
        '{"tool": "adjust_plasma_shields", "arguments": {"intensity": 85}}',
        '{"tool": "adjust_plasma_shields", "arguments": {"intensity": -20}}',   # ValueError in tool
        '{"tool": "adjust_plasma_shields", "arguments": {"intensity": "high"}}', # schema type error
        '{"tool": "navigate_warp_vector", "arguments": {"x": 1.0, "y": 2}}',     # missing z
        '{"tool": "navigate_warp_vector", "arguments": {"x": -0.85, "y": 0.52, "z": 0.05}}',
        '{"tool": "query_sensor_array", "arguments": {"target": "escape_corridor", "mode": 1}}',
        '{"tool": "self_destruct", "arguments": {}}',
        '{"tool": "query_sensor_array", "arguments": {"target": ',               # broken JSON
        '[1, 2, 3]',
    ]
    for c in cases:
        print(f"IN : {c}\nOUT: {d.dispatch(c)}\n")
