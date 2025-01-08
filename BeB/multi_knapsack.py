class Node():
    def __init__(self, UB):
        self.UB = UB

    def __ge__(self, other):
        """
        Compare two nodes based on their upper bound. We are using a min heap, so I want to process first the one with the largest Upper Bound
        """
        return self.UB >= other.UB


class BranchAndBound():
    def __init__(self):
        self.best_profit = 0
        self.best_assignment = None
        self.best_node = None