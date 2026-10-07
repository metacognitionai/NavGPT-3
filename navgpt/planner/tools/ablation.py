"""Controlled-study interventions: display interface, tool set, and the Planner
review interval. No provider or simulator dependencies."""
import threading
import time

INTERFACES = ('reference', 'text_labels', 'small_digits', 'heading_up_map', 'absolute_bearings')
CAPABILITIES = {
    'core': ('navigate_by_instruction', 'observe_forward', 'observe_panorama', 'terminate_episode'),
    'spatial_memory': ('navigate_by_instruction', 'observe_forward', 'observe_panorama', 'terminate_episode',
                       'observe_map', 'navigate_to_node', 'annotate_node'),
    'full': ('navigate_by_instruction', 'observe_forward', 'observe_panorama', 'terminate_episode',
             'observe_map', 'navigate_to_node', 'annotate_node', 'navigate_relative'),
}


def validate(interface, capabilities, review_every):
    if interface not in ('standard',) + INTERFACES:
        raise ValueError('Unknown interface: ' + interface)
    if capabilities not in CAPABILITIES:
        raise ValueError('Unknown capability bundle: ' + capabilities)
    if isinstance(review_every, bool) or not isinstance(review_every, int) or review_every < 0:
        raise ValueError('review_every must be a non-negative number of VLA steps')


class ReviewScheduler:
    """Synchronous, single-writer handoffs at committed inference boundaries.

    No background inference or queued physical commands. A suspension is reached
    only after act + execution have returned. Failed calls never advance cycles.
    Tool serialization holds the same lease through prediction and execution.
    """
    def __init__(self, review_every=0, clock=time.monotonic):
        """review_every: committed VLA inference steps between Planner reviews;
        0 reviews only when the VLA stops or reaches its step cap."""
        self.interval = review_every
        self.clock = clock
        self.lock = threading.RLock()
        self.generation = 0
        self.delegation = 0
        self.state_version = 0
        self.route = None
        self.suspended = False
        self.used = 0
        self.blocked_streak = 0
        self.review_pending = False
        self.checks = self.natural_returns = self.interruptions = 0
        self.cycles = 0
        self.total_cycles = 0
        self.wait_s = 0.0
        self.pause_time = None
        self.failed = False
        self.started = clock()
        self.trace = []
        self.travelled = 0.0

    def begin(self, route):
        if self.failed:
            raise RuntimeError('Review session invalid after uncertain inference; reset episode')
        continuing = self.suspended and self.route == route
        if self.pause_time is not None:
            self.wait_s += self.clock() - self.pause_time
            self.pause_time = None
        if self.suspended and not continuing:
            self.interruptions += 1
        if not continuing:
            self.delegation += 1
            self.used = 0
            self.trace = []
            self.travelled = 0.0
            self.blocked_streak = 0
            self.review_pending = False
        self.review_pending = False
        self.route = route
        self.suspended = False
        self.cycles = 0
        self.generation += 1
        return self.generation

    def commit(self, generation, blocked=False, natural=False):
        if generation != self.generation or self.suspended:
            raise RuntimeError('Stale or suspended motion grant')
        self.state_version += 1
        self.used += 1
        self.total_cycles += 1
        self.cycles += 1
        self.blocked_streak = self.blocked_streak + 1 if blocked else 0
        if not blocked:
            self.review_pending = False
        if self.blocked_streak == 2:
            self.review_pending = True
        if natural:
            self.natural_returns += 1
            self.generation += 1
            self.route = None
            return 'natural'
        if self.interval and self.cycles >= self.interval:
            self.checks += 1
            self.suspended = True
            self.generation += 1
            self.pause_time = self.clock()
            return 'scheduled'
        return None

    def finish(self, generation):
        if generation != self.generation or self.suspended:
            raise RuntimeError('Stale or suspended motion grant')
        self.natural_returns += 1
        self.generation += 1
        self.route = None
        return 'natural'

    def fail(self):
        self.failed = True
        self.generation += 1

    def correction(self):
        if self.pause_time is not None:
            self.wait_s += self.clock() - self.pause_time
            self.pause_time = None
        if self.suspended:
            self.interruptions += 1
            self.suspended = False
            self.route = None
        self.generation += 1
        self.state_version += 1

    def snapshot(self):
        elapsed = self.clock() - self.started
        return {'invalid': self.failed, 'retry_s': 0.0, 'completed_inference_cycles': self.total_cycles, 'state_version': self.state_version, 'delegation_id': self.delegation,
                'generation': self.generation, 'periodic_checks': self.checks,
                'natural_returns': self.natural_returns, 'accepted_interruptions': self.interruptions,
                'review_requested': self.review_pending, 'suspended': self.suspended,
                'planner_wait_s': self.wait_s, 'checks_per_second': self.checks / elapsed if elapsed else 0}
