from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class AveragedMetric(Metric):
    def __init__(self):
        self.history = []

    def update(self, value):
        self.history.append(value)

    def compute(self):
        return np.average(self.history)

    def reset(self):
        self.history.clear()


class WeightedAveragedMetric(Metric):
    def __init__(self):
        self.history = []
        self.weights = []

    def update(self, value, weight):
        self.history.append(value)
        self.weights.append(weight)

    def compute(self):
        return np.average(self.history, weights=self.weights)

    def reset(self):
        self.history.clear()
        self.weights.clear()


class Accuracy(Metric):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, correct, total):
        self.correct += correct
        self.total += total

    def compute(self):
        return self.correct / self.total

    def reset(self):
        self.correct = 0
        self.total = 0
