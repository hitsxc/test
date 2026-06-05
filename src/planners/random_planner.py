import numpy as np
from .base_planner import BasePlanner

class RandomPlanner(BasePlanner):
    def plan(self, current_state=None) -> int:
        unvisited = self.get_unvisited()
        if len(unvisited) == 0:
            return -1
        return int(np.random.choice(unvisited))
