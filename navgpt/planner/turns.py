"""Count Planner responses separated by environment feedback, across providers.

The initial response to the episode input counts once. After each completed
environment-result batch, the next nonempty Planner response counts once.
Text/thinking/tool blocks in that response are grouped; parallel tool calls and
their results do not multiply turns. A final response to STOP's result counts.
Token accounting, streaming chunks, and activity while awaiting tools do not.
"""
from pathlib import Path
import json

TURN_UNIT = 'environment_feedback_response_v1'


class PlannerTurnCounter:
    def __init__(self):
        self.count = 0
        self.awaiting_response = True
        self.pending = set()

    def observe(self, kind, payload):
        if kind == 'tool_result':
            self.pending.discard(payload.get('tool_use_id'))
            if not self.pending:
                self.awaiting_response = True
            return
        meaningful = kind == 'tool_use' or (
            kind in ('assistant_text', 'thinking') and bool((payload.get('text') or '').strip()))
        if meaningful and self.awaiting_response and not self.pending:
            self.count += 1
            self.awaiting_response = False
        if kind == 'tool_use':
            self.pending.add(payload.get('id'))


def count_turns(events):
    counter = PlannerTurnCounter()
    for event in events:
        counter.observe(event.get('kind'), event)
    return counter.count


def recorded_turns(path, metrics=None):
    """Turn count from an episode trace; a missing or replaced trace counts nothing."""
    path = Path(path)
    if not path.is_file():
        return None
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if metrics is not None and not any(e.get('kind') == 'episode_metrics' and e.get('metrics') == metrics for e in events):
        return None
    return count_turns(events)
