"""
Here the physical state is represented by numpy float 1d array.
An additional Compiler could be used to compile actions and whatsoever.
"""
import numpy as np


class BeliefNumAction:
    def __init__(self, masks, values, preconditions):
        self.masks = masks
        self.values = values
        self.precs = preconditions

    def possible_states(self, bss):
        assert isinstance(bss, NumBSS)
        sel = np.ones(bss.bss.shape[0], dtype=np.bool)
        for prec in self.precs:
            sel *= prec(bss.bss)
        return sel

class AndCondition:
    def __init__(self, values):


class NumBSS:
    def __init__(self, state):
        self.bss = None
        if isinstance(state, NumBSS):
            self.bss = state.bss
        if isinstance(state, np.ndarray) and len(state.shape) == 2:
            self.bss = state
        else:
            self.bss = np.array([state])

    def apply(self, action: BeliefNumAction, selector):
        assert isinstance(action, BeliefNumAction)
        states_selected = np.sum(selector)
        nbss = np.tile(self.bss[selector], (action.masks.shape[0], 1))
        mask = np.repeat(action.masks, states_selected, axis=0)
        act = np.repeat(action.values, states_selected, axis=0)
        nbss = np.where(mask, act, nbss)
        print(nbss)
        nbss[:, 0] = nbss[:, 0] * act[:, 0]
        return NumBSS(nbss)

    def apply_all(self, action: BeliefNumAction):
        return self.apply(action, action.possible_states(self))

    def apply_whether(self):


class Compiler:
    def __init__(self):
        self.dict = {'prob': 0}
        self.uuid = 1
        self._size = None
        self.revdict = ['prob']

    def add_var(self, name):
        assert self._size is None
        if isinstance(name, list):
            for n in name:
                self.add_var(n)
            return
        assert isinstance(name, str)
        self.dict[name] = self.uuid
        self.revdict.append(name)
        self.uuid += 1

    def fix_model(self):
        self._size = self.uuid

    def size(self):
        assert self._size is not None
        return self._size

    def make_belief_num_action(self, action, preconditions):
        masks = np.zeros((len(action), self.size()), dtype=np.bool)
        values = np.zeros((len(action), self.size()), dtype=np.float)
        for i, outcome in enumerate(action):
            for var, val in outcome.items():
                masks[i][self.dict[var]] = True
                values[i][self.dict[var]] = val
        masks[:, 0] = False
        return BeliefNumAction(masks, values, preconditions)

    def make_equality_condition(self, var, val):
        index = self.dict[var]
        return lambda s: s[:, index] == val

    def make_initial_state(self, variables: dict):
        variables.update({'prob': 1})
        state = [variables[self.revdict[i]] if self.revdict[i] in variables else 0 for i in range(self.size())]
        return NumBSS(np.array(state))


if __name__ == "__main__":
    comp = Compiler()
    comp.add_var(['a', 'b', 'c'])
    comp.fix_model()
    c1 = comp.make_equality_condition('a', 0)
    c2 = comp.make_equality_condition('a', 1)
    a1 = comp.make_belief_num_action([
        {
            'a': 1,
            'prob': 0.5
        },
        {
            'a': 2,
            'prob': 0.5
        }
    ], [c1])
    a2 = comp.make_belief_num_action([
        {
            'b': 0,
            'prob': 0.6
        },
        {
            'b': 1,
            'prob': 0.4
        }
    ], [c2])

    ins = comp.make_initial_state({'c': 3})
    print(ins.bss)
    print(ins.apply_all(a1).apply_all(a2).bss)
