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

    def prob(self, test_function_or_dict=None):
        if test_function_or_dict is None:
            return sum([p for s, p in self.states])
        else:
            if isinstance(test_function_or_dict, dict):
                items = test_function_or_dict.items()
                return sum([p for s, p in self.states if all(k in s and s[k] == v for k, v in items)])
            else:
                return sum([p for s, p in self.states if test_function_or_dict(s)])


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
        return self

    def __eq__(self, other):
        if len(self.states) != len(other.states):
            return False
        self_matched = [False for _ in self.states]
        other_matched = [False for _ in other.states]
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

        return type(self)(BeliefStateSimple._overwrite_ls_by_rs(self, other).states)

    def __or__(self, other):
        return type(self)(copy.deepcopy(self.states) + copy.deepcopy(other.states)).simplify()

    def __mul__(self, other):
        """
        applying an action which results in belief (sub)state other
        :param other: belief state
        :return: belief state
        """
        return BeliefStateSimple._overwrite_ls_by_rs(self, other)

    def __add__(self, other):
        return type(self)(self.states + other.states)

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
        return type(self)(copy.deepcopy((self // other).states))

    def __floordiv__(self, other):
        """
        selecting a subset whether other states is true and returns a linked object
        :param other: belief state
        :return: belief state
        """

        def selector(s):
            return BeliefStateSimple._has_substate_from(s, other)

        return self.select_whether(selector)

    def select_whether(self, test_function):
        """

        :param test_function: accepts physical state s.
        If test_function returns True, state will be included into selection
        :return: BeliefStateSimple
        """
        return type(self)([(s, p) for s, p in self.states if test_function(s)])

    def split_by(self, test_function):
        """

        :param test_function: function that returns True or False
        :return: (true_states, false_states)
        """

        return (
            self.select_whether(test_function),
            self.select_whether(lambda s: not test_function(s))
        )

    def apply_function(self, function):
        return [(function(s), p) for s, p in self.states]

    def bucketize(self, function):
        results = {}
        for s, p in self.states:
            r = function(s)
            if r in results:
                results[r].append((s, p))
            else:
                results[r] = [(s, p)]
        return ((result, type(self)(states)) for result, states in results.items())

    def __repr__(self):
        return self.states.__repr__()

    def apply(self, physical_state):
        for ls, lp in self.states:
            ls.update(physical_state)
        self.simplify()
        return self

    def __deepcopy__(self, memo):
        return type(self)(copy.deepcopy(self.states))

    def __copy__(self):
        return type(self)(copy.copy(self.states))
