import pandas as pd

class Permutator(object):
    def __init__(self):
        import itertools
        x = itertools.permutations([0, 1, 2])
        self.permutations = pd.Series(list(x))
        self.total_permutations = len(self.permutations)
        self.cur_permutation_index = 0

    def get_permuration_series(self):
        return pd.Series(self.permutations[self.cur_permutation_index])

    def set_next_permutation(self):
        self.cur_permutation_index += 1

    def has_more_permurations(self):
        return self.cur_permutation_index < self.total_permutations
