"""
Assuming that physical state is described by a dict()
"""
import copy
import deepdiff


class BeliefStateSimple:
    def __init__(self, physical_state, prob=1):
        if isinstance(physical_state, dict):
            self.states = [
                (physical_state, prob)
            ]
        elif isinstance(physical_state, list):
            # assuming we passed all da self.states
            self.states = [(ps, pr * prob) for ps, pr in physical_state]
        else:
            raise RuntimeWarning("physical_state argument should be either list or dict!")

    @staticmethod
    def _physical_state_equals(a, b):
        return len(deepdiff.DeepDiff(a, b)) == 0

    @staticmethod
    def _physical_state_intersects(a, b):
        return len(set(a.keys()).intersection(set(b.keys()))) != 0

    def _spaces_intersects(self, other):
        self_keys = set()
        self_keys = self_keys.union(*[s.keys() for s, p in self.states])
        other_keys = set()
        other_keys = other_keys.union(*[s.keys() for s, p in other.states])

        return len(self_keys.intersection(other_keys)) != 0

    def normalize(self, prob=1):
        old_prob = sum([p for s, p in self.states])
        mul = prob / old_prob
        self.states = [(s, p * mul) for s, p in self.states]

    def simplify(self):
        buckets = []
        for i, (state, prob) in enumerate(self.states):
            is_different = True
            for s, l in buckets:
                if BeliefStateSimple._physical_state_equals(s, state):
                    l.append(i)
                    is_different = False
                    break
            if is_different:
                buckets.append((state, [i]))
        self.states = [(s, sum([self.states[i][1] for i in l])) for s, l in buckets]

    def __eq__(self, other):
        if len(self.states) != len(other.states):
            return False
        self_matched = [False for s in self.states]
        other_matched = [False for s in other.states]
        for i, (s, p) in enumerate(self.states):
            for j, (os, op) in enumerate(other.states):
                if not self_matched[i] and not other_matched[j] and op == p and self._physical_state_equals(s, os):
                    self_matched[i] = True
                    other_matched[j] = True
        return False not in self_matched

    @staticmethod
    def _overwrite_ls_by_rs(lbs, rbs):
        new_states = []

        for ls, lp in lbs.states:
            for rs, rp in rbs.states:
                lsc = copy.deepcopy(ls)
                rsc = copy.deepcopy(rs)
                lsc.update(rsc)
                new_states.append((lsc, lp * rp))

        new_belief_state = BeliefStateSimple(new_states)
        new_belief_state.simplify()
        return new_belief_state

    def __and__(self, other):
        if self._spaces_intersects(other):
            raise RuntimeWarning("you are making an AND product of two states with non-zero intersection!")

        return BeliefStateSimple._overwrite_ls_by_rs(self, other)

    def __or__(self, other):
        return BeliefStateSimple(copy.deepcopy(self.states) + copy.deepcopy(other.states))

    def __mul__(self, other):
        """
        applying an action which results in belief (sub)state other
        :param other: belief state
        :return: belief state
        """
        return BeliefStateSimple._overwrite_ls_by_rs(self, other)

    @staticmethod
    def _has_substate_from(lps, rbs):
        lps_keys = set(lps.keys())
        for rs, rp in rbs.states:
            rs_keys = set(rs.keys())
            if lps_keys.issuperset(rs_keys):
                diff = deepdiff.DeepDiff({k: v for k, v in lps.items() if k in rs_keys}, rs)
                if len(diff) == 0:
                    return True
        return False

    def __truediv__(self, other):
        """
        selecting a subset whether other states is true and returns a copied
        :param other: belief state
        :return: belief state
        """
        return BeliefStateSimple(copy.deepcopy((self // other).states))

    def __floordiv__(self, other):
        """
        selecting a subset whether other states is true and returns a linked object
        :param other: belief state
        :return: belief state
        """
        return BeliefStateSimple([(s, p) for s, p in self.states if BeliefStateSimple._has_substate_from(s, other)])

    def __repr__(self):
        return self.states.__repr__()

    def apply(self, physical_state):
        for ls, lp in self.states:
            ls.update(physical_state)
        self.simplify()
        return self


