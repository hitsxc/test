from abc import ABC, abstractmethod

class BasePlanner(ABC):
    def __init__(self, view_space, config=None):
        self.view_space = view_space
        self.config = config or {}
        self.visited = set()

    @abstractmethod
    def plan(self, current_state=None) -> int:
        pass

    def reset(self):
        self.visited.clear()

    def mark_visited(self, idx: int):
        self.visited.add(idx)

    def get_unvisited(self):
        return [i for i in range(len(self.view_space)) if i not in self.visited]
