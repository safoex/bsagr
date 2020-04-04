from bsagr import AOBS
import random
from gi.repository import Gtk
from dd.autoref import BDD
import functools, operator, itertools


class RandomExploration:
    def __init__(self, total_vars=10, total_actions=10, max_value=10, with_bdd=False):
        self.total_vars = total_vars
        self.total_actions = total_actions
        self.aobs = AOBS()
        self.max_value = max_value
        self.bdd = BDD()
        self.bdd_expr = None
        self.bdd_var_names = [self.var_name(i) for i in range(self.total_vars)]
        self.with_bdd = with_bdd
        self.steps_made = 0
        self.belief_state_size_threshold = 1e6
        self.sizes_result = []
        # TODO: 0 <= values < self.max_value !

    @staticmethod
    def var_name(i):
        if i < 26:
            return chr(ord('a')+i)
        else:
            return RandomExploration.var_name(i // 26) + RandomExploration.var_name(i % 26)

    def initialize_aobs(self, physical_state):
        self.aobs.root = self.aobs.hash_recursive(['a'] + [[i, v] for i, v in enumerate(physical_state)])

    def act_bdd(self, preconditions, action, bdd=None):
        bdd_precs = self.assignment_bdd(*zip(*preconditions))
        # print(bdd_precs.to_expr())
        # poss, negs = self.assignment_bdd_(*zip(*preconditions))
        # print("only poss or negs: ", (self.bdd_expr & poss).dag_size, (self.bdd_expr & negs).dag_size)
        # print("before prec size: ", self.bdd_expr.dag_size)
        # print("after prec size: ", (self.bdd_expr & bdd_precs).dag_size)
        if bdd is None:
            self.bdd_expr = (self.bdd_expr & ~bdd_precs) | self.act_on_bdd(self.bdd_expr & bdd_precs, action)
        else:
            return self.act_on_bdd(bdd & bdd_precs, action)

    def act_on_bdd(self, bdd, action):
        variables = list(zip(*action[1][1][1:]))[0]
        bdd_effects = (self.assignment_bdd(*zip(*effect[1:])) for _, effect in action[1:])
        bdd_action = functools.reduce(operator.or_, bdd_effects)
        # print(variables)
        # print("size before: ", bdd.dag_size)
        bdd_cleaned = self.clean_vars_bdd(bdd, variables)
        # print("cleaned size: ", bdd_cleaned.dag_size)
        # print("after action size: ", (bdd_cleaned & bdd_action).dag_size)
        return bdd_cleaned & bdd_action

    def clean_vars_bdd(self, bdd, variables):
        # print("to clean: ", variables)
        bvars = (self.bdd_var_names[var] + str(val) for var, val in itertools.product(variables, range(self.max_value)))
        for bv in bvars:
            bdd = self.bdd.let({bv: True}, bdd) | self.bdd.let({bv: False}, bdd)
        return bdd

    def assignment_bdd(self, vars, vals):
        kvs = {v: u for v, u in zip(vars, vals)}
        # print("vars: ", vars)
        # print("vals: ", vals)

        bvars = list(self.bdd_var_names[var] + str(val) for var, val in zip(vars, vals))
        # print(bvars)
        positives = functools.reduce(operator.and_, (self.bdd.add_expr(bvar) for bvar in bvars))
        negatives_bvars = list(
            self.bdd_var_names[var] + str(val) for var, val in itertools.product(vars, range(self.max_value)) if
            kvs[var] != val)
        # print(negatives_bvars)
        negatives = functools.reduce(operator.and_, (~self.bdd.add_expr(bvar) for bvar in negatives_bvars))
        return positives & negatives

    def assignment_bdd_(self, vars, vals):
        kvs = {v: u for v, u in zip(vars, vals)}
        # print("vars: ", vars)
        # print("vals: ", vals)

        bvars = list(self.bdd_var_names[var] + str(val) for var, val in zip(vars, vals))
        # print(bvars)
        positives = functools.reduce(operator.and_, (self.bdd.add_expr(bvar) for bvar in bvars))
        negatives_bvars = list(
            self.bdd_var_names[var] + str(val) for var, val in itertools.product(vars, range(self.max_value)) if
            kvs[var] != val)
        # print(negatives_bvars)
        negatives = functools.reduce(operator.and_, (~self.bdd.add_expr(bvar) for bvar in negatives_bvars))
        return positives, negatives

    def initialize_bdd(self, physical_state):
        for var, val in itertools.product(self.bdd_var_names, range(self.max_value)):
            self.bdd.add_var(var + str(val))
        self.bdd_expr = self.assignment_bdd(*zip(*enumerate(physical_state)))

    def random_init(self):
        ps = [random.randint(0, self.max_value - 1) for _ in range(self.total_vars)]
        self.initialize_aobs(ps)
        if self.with_bdd:
            self.initialize_bdd(ps)
        return ps

    def generate_random_action(self, precs, total, each):
        preconds = {}
        while len(preconds) < precs:
            preconds[random.randint(0, self.total_vars - 1)] = random.randint(0, self.max_value - 1)
        preconditions = [(var, lambda v, val=val: v == val) for var, val in preconds.items()]
        preconditions_bdd = list(preconds.items())
        eff_vars = set()
        while len(eff_vars) < each:
            eff_vars.add(random.randint(0, self.total_vars - 1))
        probs = [random.random() for _ in range(total)]
        norm = sum(probs)
        probs = [p / norm for p in probs]
        action = ['o'] + [
            (p, ['a'] + [[v, random.randint(0, self.max_value - 1)] for v in eff_vars])
            for p in probs
        ]
        return preconditions, action, preconditions_bdd

    def random_exploration(self, steps=20, precs=3, effects=3, each=2, each_step=False):
        initial_state = self.random_init()
        actions_applied = []
        j = 0
        self.steps_made = 0
        while len(actions_applied) < steps:
            preconditions, action, preconditions_bdd = self.generate_random_action(precs, effects, each)
            if self.aobs.prob(preconditions) > 0:
                # print("aobs nodes before: ", self.aobs.dag_size())
                self.aobs.act(preconditions, action)
                # print("aobs nodes after: ", self.aobs.dag_size())
                # print("aobs states: ", self.aobs.count_physical_states_real())
                if self.with_bdd:
                    self.act_bdd(preconditions_bdd, action)
                actions_applied.append((preconditions_bdd, action))
                j = 0
                self.steps_made = len(actions_applied)
                if each_step:
                    self.sizes_result.append(dict(total=self.total_vars * self.aobs.count_physical_states_real(),
                    aobs=self.aobs.dag_size(),
                    aobs_greedy=self.aobs.size_after_greedy(include_prob=True),
                    total_naive_dup=self.aobs.estimate_number_of_states() * self.total_vars,
                    bdd=self.bdd_expr.dag_size
                         ))
                if self.aobs.estimate_number_of_states() * self.total_vars > self.belief_state_size_threshold:
                    return True
                # print(len(actions_applied))
            else:
                j += 1
                if j > 10000:
                    return False
        return True


from joblib import Parallel, delayed
import pickle, time

if __name__ == "__main__":
    def test():
        steps = 40
        re = RandomExploration(total_vars=40, max_value=3)
        while not re.random_exploration(steps):
            pass
        # print()
        # print (_steps)
        # print(len(re.aobs.as_collection())*re.total_vars)
        return len(re.aobs.as_collection()) * re.total_vars, sum(len(n) for _, n in re.aobs.nodes.items())
        # print(len(re.aobs.nodes)*2)


    # res = list(Parallel(n_jobs=11)(delayed(test)() for i in range(200)))
    # with open("steps=40total_vars=40max_value=2t="+ str(time.time()) + ".pydump",'wb') as f:
    #     pickle.dump(res, f)
    z = 0
    def test_bdd(total_vars, steps, max_value, effects=3, vars_in_each_effect=3, preconditions=3):
        try:
            re = RandomExploration(total_vars=total_vars, max_value=max_value, with_bdd=True)
            while not re.random_exploration(steps, precs=preconditions, effects=effects, each=vars_in_each_effect):
                pass
            re.bdd.collect_garbage()
            sizes = dict(total=re.total_vars * re.aobs.count_physical_states_real(),
                    aobs=re.aobs.dag_size(),
                    aobs_greedy=re.aobs.size_after_greedy(include_prob=True),
                    total_naive_dup=len(re.aobs.as_collection()) * re.total_vars,
                    bdd=re.bdd_expr.dag_size
                         )
            global z
            z += 1
            print(sizes)
            return dict(
                total_vars=re.total_vars,
                max_value=re.max_value,
                steps=re.steps_made,
                preconditions=preconditions,
                effects=effects,
                vars_in_each_effect=vars_in_each_effect,
                sizes=sizes
            )
        except BaseException as e:
            return {}
    def test_bdd_steps(total_vars, steps, max_value, effects=3, vars_in_each_effect=3, preconditions=3):
        try:
            re = RandomExploration(total_vars=total_vars, max_value=max_value, with_bdd=True)
            while not re.random_exploration(steps, precs=preconditions, effects=effects, each=vars_in_each_effect, each_step=True):
                pass
            re.bdd.collect_garbage()
            sizes = dict(total=re.total_vars * re.aobs.count_physical_states_real(),
                    aobs=re.aobs.dag_size(),
                    aobs_greedy=re.aobs.size_after_greedy(include_prob=True),
                    total_naive_dup=len(re.aobs.as_collection()) * re.total_vars,
                    bdd=re.bdd_expr.dag_size
                         )
            global z
            z += 1
            print(sizes)

            return [dict(
                total_vars=re.total_vars,
                max_value=re.max_value,
                steps=steps,
                preconditions=preconditions,
                effects=effects,
                vars_in_each_effect=vars_in_each_effect,
                sizes=sz
            ) for steps, sz in enumerate(re.sizes_result)]
        except BaseException as e:
            return {}

    # print(test_bdd(steps=20, total_vars=40, max_value=4, effects=5, vars_in_each_effect=8)['sizes'])




    t = str(time.time())

    experiment1 = sum(list(Parallel(n_jobs=8)(
        delayed(test_bdd_steps)(steps=20, total_vars=40, max_value=2) for i in range(200))), [])
    print("experiment1 done")
    with open("experiment1_" + t + ".nds", 'wb') as f:
        pickle.dump(experiment1, f)

    # # steps dependency
    # experiment1 = list(Parallel(n_jobs=11)(delayed(test_bdd)(steps=steps,total_vars=40, max_value=max_value) for i, steps, max_value in
    #                                           itertools.product(range(100), range(5, 35, 2), [2, 4, 8])))
    # print("experiment1 done")
    # with open("experiment1_" + t + ".nds", 'wb') as f:
    #     pickle.dump(experiment1, f)
    #
    # # vars number and limit dependency
    # experiment2 = list(Parallel(n_jobs=11)(delayed(test_bdd)(steps=20,total_vars=total_vars, max_value=max_value) for i, total_vars, max_value in
    #                                           itertools.product(range(100), range(5, 35, 5), [2, 4, 8])))
    #
    # print("experiment2 done")
    # with open("experiment2_" + t + ".nds", 'wb') as f:
    #     pickle.dump(experiment2, f)
    #
    # # action properties dependency
    # experiment3 = list(Parallel(n_jobs=11)(
    #     delayed(test_bdd)(steps=20, total_vars=40, max_value=4, effects=effects, vars_in_each_effect=vars_in_each) for i, effects, vars_in_each in
    #     itertools.product(range(100), range(2, 4), [1, 2, 4, 8])))
    #
    # print("experiment3 done")
    # with open("experiment3_" + t + ".nds", 'wb') as f:
    #     pickle.dump(experiment2, f)

# t['re'].aobs.draw()
    # Gtk.main()

    # bdd = BDD()
    #
    # bdd.declare('a0', 'b0', 'b1', 'a1', 'c0', 'c1')
    # a0, a1 = bdd.add_expr('a0'), bdd.add_expr('a1')
    # b0, b1 = bdd.add_expr('b0'), bdd.add_expr('b1')
    # c0, c1 = bdd.add_expr('c0'), bdd.add_expr('c1')
    # _states = [[a, b, c, 1 - a, 1 - b, 1 - c] for a, b, c in itertools.product(range(2), range(2), range(2))]
    #
    # vs = ['a0', 'b0', 'c0', 'a1', 'b1', 'c1']
    # states = [{k: state[i] == 1 for i, k in enumerate(vs)} for state in _states]
    #
    # s = a0 & b0 & c0 & (~a1 & ~b1 & ~c1)
    # s0 = s
    # c = a0 & ~a1
    # a = b0 & ~b1 | b1 & ~b0
    # re = RandomExploration()
    # z = s & c
    # for bv in ['b0', 'b1']:
    #     z = bdd.let({bv: True}, z) | bdd.let({bv: False}, z)
    # z = z & a
    # s1 = s0 & ~c | z
    #
    # # for state in states:
    # #     print(state)
    # #     print(bdd.let({v: state[v] for v in vs}, s1).to_expr())
    #
    # C1 = a0 & ~a1 & b0 & ~b1
    # a = c1 & ~c0 | ~c1 & c0
    # z = s1 & C1
    # for bv in ['c0', 'c1']:
    #     z = bdd.let({bv: True}, z) | bdd.let({bv: False}, z)
    # z1 = z & a
    # s2 = s1 & ~C1 | z1
    # bdd.dump("awesome.pdf", roots=[z & a])
    # for state in states:
    #     print(state)
    #     print(bdd.let({v: state[v] for v in vs}, s2).to_expr())
