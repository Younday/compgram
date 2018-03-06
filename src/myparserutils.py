"""

An implementation of a greedy transition-based dependency parser (unlabeled parsing only).
Released under BSD license.

Code is an adapted version of Matthew Honnibal's parser:
https://explosion.ai/blog/parsing-english-in-python

-- change: move core logic to separate myparserutils file

modified by bplank, 03/2017

"""
#### Helper classes - do not modify ####
import os
from collections import defaultdict
import pickle

## Global
SHIFT = 0; RIGHT = 1; LEFT = 2
MOVES = (SHIFT, RIGHT, LEFT)


class DefaultList(list):
    """A list that returns a default value if index out of bounds."""
    def __init__(self, default=None):
        self.default = default
        list.__init__(self)

    def __getitem__(self, index):
        try:
            return list.__getitem__(self, index)
        except IndexError:
            return self.default


class Parse(object):
    """
    Structure that keeps current set of edges/arcs A
    lefts: left-branching edges
    rights: right-branching edges
    """
    def __init__(self, n):
        self.n = n
        self.heads = [None] * (n-1)
        self.labels = [None] * (n-1)
        self.lefts = []
        self.rights = []
        for i in range(n+1):
            self.lefts.append(DefaultList(0))
            self.rights.append(DefaultList(0))

    def add(self, head, child, label=None):
        self.heads[child] = head
        self.labels[child] = label
        if child < head:
            self.lefts[head].append(child)
        else:
            self.rights[head].append(child)

#### End helper classes ####



class Perceptron(object):
    """
    Learn weights for the features using the Perceptron algorithm
    """
    def __init__(self, classes=None):
        self.classes = classes
        # Each feature gets its own weight vector, so weights is a dict-of-arrays
        self.weights = {}
        # The accumulated values, for the averaging. These will be keyed by
        # feature/clas tuples
        self._totals = defaultdict(int)
        # The last time the feature was changed, for the averaging. Also
        # keyed by feature/clas tuples
        # (tstamps is short for timestamps)
        self._tstamps = defaultdict(int)
        # Number of instances seen
        self.i = 0

    def predict(self, features):
        '''Dot-product the features and current weights and return the best class.'''
        scores = self.score(features)
        # Do a secondary alphabetic sort, for stability
        return max(self.classes, key=lambda clas: (scores[clas], clas))

    def score(self, features):
        """
        Scores = features \cdot weights
        """
        all_weights = self.weights
        scores = dict((clas, 0) for clas in self.classes)
        for feat, value in features:
            if value == 0:
                continue
            if feat not in all_weights:
                continue
            weights = all_weights[feat]
            for clas, weight in list(weights.items()):
                scores[clas] += value * weight
        return scores

    def update(self, truth, guess, features):
        """
        Update parameters
        """
        def upd_feat(c, f, w, v):
            param = (f, c)
            self._totals[param] += (self.i - self._tstamps[param]) * w
            self._tstamps[param] = self.i
            self.weights[f][c] = w + v

        self.i += 1
        if truth == guess:
            return None
        for f, val in features:
            weights = self.weights.setdefault(f, {})
            upd_feat(truth, f, weights.get(truth, 0.0), 1.0)
            upd_feat(guess, f, weights.get(guess, 0.0), -1.0)

    def average_weights(self):
        for feat, weights in list(self.weights.items()):
            new_feat_weights = {}
            for clas, weight in list(weights.items()):
                param = (feat, clas)
                total = self._totals[param]
                total += (self.i - self._tstamps[param]) * weight
                averaged = round(total / float(self.i), 3)
                if averaged:
                    new_feat_weights[clas] = averaged
            self.weights[feat] = new_feat_weights

    def save(self, path):
        print("Saving model to %s" % path)
        pickle.dump(self.weights, open(path, 'wb'))

    def load(self, path):
        self.weights = pickle.load(open(path))




