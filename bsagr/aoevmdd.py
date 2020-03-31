import copy
import xdot

from gi.repository import Gtk
import functools, itertools
import operator
from queue import PriorityQueue


class AOBS:
    def __init__(self):
        """
        descriptive example of empty constructor
        """
        self.nodes = {}
        self.vars = {}
        self.parents = {}
        self.hash_limit = 2
        self.hash_seeds = [1343452, 12432]
        self.root = self.hash_recursive([0, 0.0])
        self.colors = ['green', 'red', 'purple']
        self.precision = 3
        self.windows = []

    @staticmethod
    def hname(h):
        return "L" + (str(h) if h >= 0 else "0" + str(abs(h)))

    @staticmethod
    def get_var_name(v):
        if v == 0:
            return 'a'
        else:
            s = ""
            while v > 0:
                s += chr(ord('a') + v % 26)
                v //= 26
            return s

    def get_color(self, h, colors):
        if colors is not None and h in colors:
            if colors[h] is True:
                return self.colors[0]
            elif colors[h] is False:
                return self.colors[1]
            else:
                return self.colors[2]
        else:
            return "black"

    def req_dot_str_ver(self, h, colors, drawn):
        if h in drawn:
            return ""
        drawn[h] = True
        hl = self.hname(h)
        n = self.nodes[h]
        hs = hl + "[label=\""
        if self.is_or(n):
            hs += "OR"
        if self.is_and(n):
            hs += "AND"
        if self.is_literal(n):
            hs += self.get_var_name(n[0]) + " = " + str(n[1])
        shape = "rectangle" if self.is_literal(n) else "circle"
        hs += "\",shape=%s, color=%s];\n" % (shape, self.get_color(h, colors))
        if self.is_or(n):
            for pc, hc in n[1:]:
                hs += hl + "-> " + self.hname(hc) + (" [ label=\"%." + "%d" % self.precision + "f\"] ;\n") % pc
            for _, hc in n[1:]:
                hs += self.req_dot_str_ver(hc, colors, drawn)
        if self.is_and(n):
            for hc in n[1:]:
                hs += hl + " -> " + self.hname(hc) + ";\n"
            for hc in n[1:]:
                hs += self.req_dot_str_ver(hc, colors, drawn)
        return hs

    def dot_str(self, colors=None, start=None, window_header=None):
        return \
            """digraph %s { 
                    rankdir=TD;
                    node[shape=circle];
            
            """ % ("aobs") \
            + (
                "" if window_header is None or (isinstance(window_header, int) and window_header < 0) else "AA[label=\"%s\"shape=rectangle, color=blue];\n" % str(window_header)) \
            + self.req_dot_str_ver(start or self.root, colors, {}) \
            + "\n}"

    def insert(self, h, tnode):
        if h not in self.nodes:
            self.nodes[h] = tnode
        if isinstance(tnode[0], int):
            var, val = tnode
            if var not in self.vars:
                self.vars[var] = {}
            if val not in self.vars[var]:
                self.vars[var][val] = h
        else:
            if tnode[0] == 'a':
                gen = (hc for hc in tnode[1:])
            else:
                gen = (hc for pc, hc in tnode[1:])
            for hc in gen:
                if hc not in self.parents:
                    self.parents[hc] = set()
                self.parents[hc].add(h)

    def hash(self, node, insert=True):
        tnode = tuple(node)
        hashes = [hash((tnode, hseed)) for hseed in self.hash_seeds]
        collision = [h in self.nodes and self.nodes[h] != tnode for h in hashes]
        if all(collision):
            raise RuntimeError("Ooops. %d hashes is not enough!" % self.hash_limit)
        hash_choosen = hashes[collision.index(False)]
        if insert:
            self.insert(hash_choosen, tnode)
        return hash_choosen

    def hash_recursive(self, node, insert=True):
        hnode = ()
        if self.is_literal(node):
            hnode = node
        if self.is_and(node):
            hnode = ['a'] + sorted([self.hash_recursive(n, insert) if isinstance(n, list) or isinstance(n, tuple) else n for n in node[1:]])
        if self.is_or(node):
            hnode = ['o'] + sorted(
                [(p, self.hash_recursive(n, insert)) if isinstance(n, list) or isinstance(n, tuple) else (p, n) for p, n in node[1:]])
        assert len(hnode)
        return self.hash(hnode, insert)

    def set_physical_state(self, state):
        self.root = self.hash_recursive(['a'] + list(enumerate(state)), insert=True)

    @staticmethod
    def is_literal(n):
        return isinstance(n[0], int)

    @staticmethod
    def is_and(n):
        return n[0] == 'a'

    @staticmethod
    def is_or(n):
        return n[0] == 'o'

    @staticmethod
    def set_color(h, colors, visited, color=None):
        colors[h] = color
        visited[h] = True
        return color

    def variablize(self, h, variables):
        if h in variables:
            return variables[h]
        n = self.nodes[h]
        if self.is_literal(n):
            variables[h] = {n[0]}
            return variables[h]
        if self.is_and(n):
            vs = set()
            for hc in n[1:]:
                vs.update(self.variablize(hc, variables))
            variables[h] = vs
            return vs
        if self.is_or(n):
            variables[h] = self.variablize(n[1][1], variables)
            for _, hc in n[1:]:
                self.variablize(hc, variables)
            return variables[h]
        assert False

    def colors_from_variables(self, variables: dict, required_subset: set):
        return {h: required_subset.issubset(vs) for h, vs in variables.items()}

    def find_min_clusters(self, h, variables: dict, colors: dict, required_subset: set, min_clusters: dict):
        if h in min_clusters:
            return min_clusters[h]
        n = self.nodes[h]
        if self.is_literal(n):
            min_clusters[h] = required_subset.issubset(variables[h])
            return min_clusters[h]
        if self.is_and(n):
            already = any([
                self.find_min_clusters(hc, variables, colors, required_subset, min_clusters) is not False
                for hc in n[1:]]
            )
            if already:
                min_clusters[h] = None
                return None
            else:
                min_clusters[h] = required_subset.issubset(variables[h]) and colors[h] is not False
                return min_clusters[h]
        if self.is_or(n):
            if all([
                self.find_min_clusters(hc, variables, colors, required_subset, min_clusters) is True
                for _, hc in n[1:]]
            ):
                min_clusters[h] = True
                for _, hc in n[1:]:
                    min_clusters[hc] = False
            else:
                min_clusters[h] = None if any([
                    self.find_min_clusters(hc, variables, colors, required_subset, min_clusters) is not False
                    for _, hc in n[1:]]
                ) else False
            return min_clusters[h]
        assert False

    def colorize(self, h, colors, visited):
        if h in colors:
            return self.set_color(h, colors, visited, colors[h])
        n = self.nodes[h]
        assert self.is_literal(n) or self.is_and(n) or self.is_or(n)

        if self.is_literal(n):
            return self.set_color(h, colors, visited, True)
        if self.is_and(n):
            has_mixed = False
            for hc in n[1:]:
                cc = self.colorize(hc, colors, visited)
                if cc == False:
                    return self.set_color(h, colors, visited, False)
                if cc is None:
                    has_mixed = True
            if has_mixed:
                return self.set_color(h, colors, visited, None)
            else:
                return self.set_color(h, colors, visited, True)
        if self.is_or(n):
            has_true = False
            has_false = False
            for pc, hc in n[1:]:
                cc = self.colorize(hc, colors, visited)
                if cc is False or cc is None:
                    has_false = True
                if cc is True or cc is None:
                    has_true = True
            if has_true and has_false:
                return self.set_color(h, colors, visited, None)
            return self.set_color(h, colors, visited, has_true)

    @staticmethod
    def get_next_not_visited(stack, visited):
        last = stack[-1]
        while len(stack) > 0 and last in visited:
            stack.pop(-1)
            last = stack[-1]
        if len(stack) == 0:
            return None
        else:
            stack.pop(-1)
            return last

    def get_subset_of_node(self, h):
        """
        returns a subset of variables in this subgraph. It is enough to follow just by one of OR node links each time.
        :param h: node hash
        :return: set of ints (variables)
        """
        n = self.nodes[h]
        if self.is_literal(n):
            return set(n[0])
        if self.is_or(n):
            return self.get_subset_of_node(n[1][1])
        # if self.is_and(n):
        assert self.is_and(n)
        vars = set()
        for hc in n[1:]:
            vars.update(self.get_subset_of_node(hc))
        return vars

    def colorize_and_cover(self, true_conditions: list, colors, actions, req_vars: set):
        stack = copy.deepcopy(true_conditions)
        visited = {}
        ready_clusters = []
        while (not all(h in visited for h in actions)) and (not all(h in visited for h in true_conditions)):
            h = self.get_next_not_visited(stack, visited)
            c = self.colorize(h, colors, visited)
            if c is True or c is None:
                if req_vars.issubset(self.get_subset_of_node(h)):
                    """
                    if our subgraph has both all conditions and action variables, we stop the search...
                    """
                    ready_clusters.append(h)
                else:
                    """
                    in the other case we continue moving to the root expanding by all parent links.
                    """
                    stack.extend(self.parents[h])

    def rehash_recursive(self, h, exceptions=None):
        n = self.nodes[h]
        if exceptions is not None and h in exceptions:
            return self.hash_recursive(exceptions[h])
        if self.is_literal(n):
            return self.hash_recursive(n)
        if self.is_and(n):
            return self.hash_recursive(['a'] + [self.rehash_recursive(hc, exceptions) for hc in n[1:]])
        if self.is_or(n):
            return self.hash_recursive(['o'] + [(pc, self.rehash_recursive(hc, exceptions)) for pc, hc in n[1:]])

    def replace_with_in(self, h, hnew, hp):
        n = self.nodes[hp]
        nn = [n[0]]
        if self.is_and(n):
            nn += [hc if hc != h else hnew for hc in n[1:]]
        else:
            nn += [(pc, hc) if hc != h else (pc, hnew) for pc, hc in n[1:]]
        self.nodes[hp] = nn

    def is_parent(self, hp, hc):
        n = self.nodes[hp]
        if self.is_and(n):
            return any((h == hc for h in n[1:]))
        if self.is_or(n):
            return any((h == hc for p, h in n[1:]))
        return False

    def replace_with(self, h, hnew):
        # self.nodes[h] = self.nodes[hnew]

        # for hp in self.nodes:
        #     if self.is_parent(hp, h):
        #         self.replace_with_in(h, hnew, hp)
        self.root = self.rehash_recursive(self.root, exceptions={h: self.nodes[hnew]})

    def isolate(self, h, colors):
        n = self.nodes[h]
        if self.is_literal(n):
            return h
        if self.is_or(n):
            if colors[h] is None:
                nn = ['o']
                for pc, hc in n[1:]:
                    nhc = self.isolate(hc, colors)
                    nnhc = self.nodes[nhc]
                    if self.is_or(nnhc):
                        nn.extend((pc * pcc, hcc) for pcc, hcc in nnhc[1:])
                    else:
                        nn.append((pc, hc))
                new_h = self.hash_recursive(nn)
                self.colorize(new_h, colors, {})
                return new_h
            else:
                return h
        if self.is_and(n):
            if colors[h] is None:
                mixed = []
                red = []
                for hc in n[1:]:
                    if colors[hc] is True:
                        red.append(hc)
                    else:
                        assert colors[hc] is None
                        mixed.append(hc)
                red_and = ['a']
                black_and = ['a']
                black_or = ['o']
                reds = []
                blacks = []

                for m in mixed:
                    mh = self.isolate(m, colors)
                    mn = self.nodes[mh]
                    assert self.is_or(mn)
                    mred = ['o']
                    mblack = ['o']
                    for pc, hc in mn[1:]:
                        if colors[hc] is True:
                            mred.append((pc, hc))
                        else:
                            mblack.append((pc, hc))
                    reds.append(mred)
                    blacks.append(mblack)
                if len(mixed) == 1:
                    black_or = blacks[0]
                    red_and.append(reds[0])
                else:
                    for _ in reds:
                        black_or.append((1, ['a']))
                    for i, r in enumerate(reds):
                        red_and.append(r)
                        for j, (_, b) in enumerate(black_or[1:]):
                            if j == i:
                                b.append(blacks[i])
                            else:
                                b.append(mixed[i])
                black_and.append(black_or)
                for r in red:
                    red_and.append(r)
                    black_and.append(r)
                big_or = ['o', (1, red_and), (1, black_and)]

                new_h = self.hash_recursive(big_or)
                self.colorize(new_h, colors, {})
                return new_h
            else:
                return h
        assert False

    def remove_vars_from(self, h, variables: set):
        n = self.nodes[h]
        if self.is_literal(n):
            if n[0] in variables:
                return None
            else:
                return h
        if self.is_and(n):
            nn = ['a']
            for hc in n[1:]:
                rhc = self.remove_vars_from(hc, variables)
                if rhc is not None:
                    nn.append(rhc)
            if len(nn) == 1:
                return None
            else:
                return self.hash_recursive(nn)
        if self.is_or(n):
            nn = ['o']
            for pc, hc in n[1:]:
                rhc = self.remove_vars_from(hc, variables)
                if rhc is not None:
                    nn.append((pc, rhc))
            if len(nn) == 1:
                return None
            else:
                return self.hash_recursive(nn)

    def act_on(self, h, an, variables: set = None):
        if variables is None:
            variables = self.variablize(self.hash_recursive(an), {})
        cleaned = self.remove_vars_from(h, variables)
        if cleaned is not None:
            return self.hash_recursive(['a', cleaned, an])
        else:
            return self.hash_recursive(an)

    def normalize(self, h):
        n = self.nodes[h]
        if self.is_literal(n):
            return 1, h
        if self.is_and(n):
            if len(n) == 2:
                return self.normalize(n[1])
            else:
                p = 1
                nn = ['a']
                for hc in n[1:]:
                    phc, newhc = self.normalize(hc)
                    p *= phc
                    nnhc = self.nodes[newhc]
                    if self.is_and(nnhc):
                        nn.extend(nnhc[1:])
                    else:
                        nn.append(newhc)
                return p, self.hash_recursive(nn)
        if self.is_or(n):
            if len(n) == 2:
                p, nn = self.normalize(n[1][1])
                return p * n[1][0], nn
            else:
                nn_ = ((pc, *self.normalize(hc)) for pc, hc in n[1:])
                nn = ['o']
                for pc, pn, nhc in nn_:
                    nnhc = self.nodes[nhc]
                    if self.is_or(nnhc):
                        nn.extend((pc * pn * pcc, hcc) for pcc, hcc in nnhc[1:])
                    else:
                        nn.append((pc * pn, nhc))
                """
                uniqueness
                """
                nd = dict()
                for pc, hc in nn[1:]:
                    if hc in nd:
                        nd[hc] += pc
                    else:
                        nd[hc] = pc
                nn = ['o'] + [(pc, hc) for hc, pc in nd.items()]
                if len(nn) == 2:
                    return nn[1]
                """
                probability normalization
                """
                norm_p = sum(pc for pc, hc in nn[1:])
                if norm_p != 1:
                    return norm_p, self.hash_recursive(['o'] + [(pc / norm_p, hc) for pc, hc in nn[1:]])
                else:
                    return 1, self.hash_recursive(nn)
        assert False

    def get_true_literals_from_conditions(self, conditionary_functions):
        true_conditions = set()
        all_colors = {}
        for var, f in conditionary_functions:
            for val in self.vars[var]:
                color = f(val)
                all_colors.update({self.vars[var][val]: color})
                if color:
                    true_conditions.add(self.vars[var][val])
        return true_conditions, all_colors

    def cleanup_rec(self, h, visited):
        n = self.nodes[h]
        visited.add(h)
        if self.is_literal(n):
            return
        if self.is_and(n):
            for hc in n[1:]:
                if hc not in self.parents:
                    self.parents[hc] = {h}
                else:
                    self.parents[hc].add(h)
                self.cleanup_rec(hc, visited)
        if self.is_or(n):
            for pc, hc in n[1:]:
                if hc not in self.parents:
                    self.parents[hc] = {h}
                else:
                    self.parents[hc].add(h)
                self.cleanup_rec(hc, visited)

    def cleanup(self):
        self.parents = {}
        visited = set()
        self.cleanup_rec(self.root, visited)
        for h in set(self.nodes.keys()).difference(visited):
            self.nodes.pop(h)

    def draw(self, colors=None, start=None, window_header=None):
        win = xdot.DotWindow()
        win.set_dotcode(bytes(self.dot_str(colors, start, window_header or len(self.windows)), 'utf-8'))
        self.windows.append(win)

    def black_colors_for_variables(self, vars_list):
        colors = {}
        for var in vars_list:
            for _, h in self.vars[var].items():
                colors[h] = False
        return colors

    def colors_for_good_literals(self, literals):
        colors = self.black_colors_for_variables(l[0] for l in literals)
        for hl in literals:
            colors[self.hash_recursive(hl)] = True
        return colors

    def act(self, cfs, action, debug_draw=False):
        action_subset = self.variablize(self.hash_recursive(action), {})
        true_literals, colors = self.get_true_literals_from_conditions(cfs)
        true_literals_initial, colors_initial = self.get_true_literals_from_conditions(cf for cf in cfs if cf[0] not in action_subset)
        req_subset = set()
        for tl in true_literals_initial:
            req_subset.add(self.nodes[tl][0])
        req_subset.update(action_subset)
        self.colorize(self.root, colors_initial, {})
        self.colorize(self.root, colors, {})
        if debug_draw: self.draw(colors=colors_initial, window_header="1\ncolorized according to conditions:\ngreen - True, red - False, purple - mixed (OR and up)")
        variables = {}
        self.variablize(self.root, variables)
        min_clusters = {}
        self.find_min_clusters(self.root, variables, colors_initial, req_subset, min_clusters)
        if debug_draw: self.draw(colors=min_clusters, window_header="2\nfound minimal clusters (green) which are subgraphs,\n where we should apply actions")
        clusters = [h for h, v in min_clusters.items() if v]
        new_clusters = [self.normalize(self.isolate(hcl, colors))[1] for hcl in clusters]
        for hcl, hclnew in zip(clusters, new_clusters):
            nhclnew = self.nodes[hclnew]
            self.colorize(hclnew, colors, {})
            if self.is_or(nhclnew):
                """
                assuming that action is OR of ANDs of literals
                """
                # if self.is_or(action):
                #     action_excluding_duplicates = ['o']
                #     local_variables = {}
                #     self.variablize(hclnew, local_variables)
                #     for pe, heand in action[1:]:
                #         effect_literals = heand[1:]
                #         action_colors = self.colors_for_good_literals(effect_literals)
                #         self.colorize(hclnew, action_colors, {})
                #         local_min_clusters = {}
                #         self.find_min_clusters(hclnew, local_variables, action_colors, action_subset, local_min_clusters)
                #         # print(local_min_clusters)
                #         # self.draw(start=hclnew, colors=local_min_clusters)

                nn = ['o'] + [(pc, self.act_on(hc, action) if colors[hc] is True else hc) for pc, hc in nhclnew[1:]]
                self.replace_with(hcl, self.hash_recursive(nn))
            if self.is_and(nhclnew):
                w = self.act_on(hclnew, action)
                self.replace_with(hcl, w)
        if debug_draw: self.draw(window_header="3\nbefore normalization procedure")
        _, self.root = self.normalize(self.root)
        self.cleanup()
        if debug_draw: self.draw(window_header="4\nnormalized\nand\ncleaned!\n:)")

    @staticmethod
    def _multiply(bs1, bs2):
        return [
            ( p1*p2, sorted(tuple(ps1 + ps2)) )
            for p2, ps2 in bs2
            for p1, ps1 in bs1
        ]

    @staticmethod
    def _simplify(bs):
        states = {}
        for p, ps in bs:
            h = hash(ps)
            if h in states:
                found = False
                for i, (hp, hps) in enumerate(states[h]):
                    if hps == ps:
                        states[h] = (p + hp, ps)
                        found = True
                if not found:
                    states[h].append((p, ps))
            else:
                states[h] = [(p, ps)]
        return [(p, ps) for _, state in states.items() for p, ps in state]

    def as_collection_rec(self, h):
        n = self.nodes[h]
        if self.is_literal(n):
            return [(1, [n])]
        if self.is_and(n):
            return functools.reduce(self._multiply, (self.as_collection_rec(hc) for hc in n[1:]))
        if self.is_or(n):
            return functools.reduce(operator.add, ([(pr * pc, hr) for pr, hr in self.as_collection_rec(hc)] for pc, hc in n[1:]))
        assert False

    def count_physical_states_real(self):
        x = AOBS()
        x.from_collection(self.as_collection())
        return sum(x.is_and(n) for _, n in x.nodes.items())

    def dag_size(self):
        return sum(2 * len(n) - 1 if self.is_or(n) else len(n) for _, n in self.nodes.items())

    def as_collection(self):
        return self.as_collection_rec(self.root)

    def from_collection(self, states):
        if len(states) > 1:
            d = ['o'] +\
                [
                    (pc, ['a'] + nc) for pc, nc in states
                ]
            self.root = self.hash_recursive(
                d
            )
        else:
            self.root = self.hash_recursive(['a'] + states[0][1])
        self.cleanup()

    def prob_rec(self, h, colors):
        n = self.nodes[h]
        if self.is_or(n):
            if colors[h] is None:
                return sum(pc * self.prob_rec(hc, colors) for pc, hc in n[1:] )
            if colors[h] is True:
                return 1
            if colors[h] is False:
                return 0
        if self.is_and(n):
            if colors[h] is False:
                return 0
            else:
                return functools.reduce(operator.mul, (self.prob_rec(hc, colors) for hc in n[1:]))
        if self.is_literal(n):
            if h not in colors:
                return 1
            elif colors[h] is False:
                return 0
            else:
                return 1
        assert False

    def prob(self, cfs):
        """
        get probability of conditions set [(var, f(var)]
        :param cfs: list of tuples (var_i, f(var_i))
        :return: 0 <= p <= 1
        """
        true_literals, colors = self.get_true_literals_from_conditions(cfs)
        self.colorize(self.root, colors, {})
        return self.prob_rec(self.root, colors)

    def compute_all_fractions_of_and(self):
        sets = {h for h, n in self.nodes.items() if self.is_and(n)}
        while True:
            to_add = set()
            for a in itertools.product(sets, sets):
                s = []
                for i in (0,1):
                    n = self.nodes[a[i]]
                    s.append( set(n[1:]))
                i_d = s[0].intersection(s[1]), s[0].symmetric_difference(s[1])
                for s in i_d:
                    a = ['a'] + list(s)
                    h = self.hash_recursive(a)
                    if h not in sets:
                        to_add.add(h)
            if len(to_add) == 0:
                break
            else:
                sets.update(to_add)
                print(len(sets), len(to_add))
        return sets

    def count_greedy_effect(self, nodes_as_ands, h_multi=1, threshold=3):
        _ands = nodes_as_ands
        initial_lens = sum((len(n) -1) * h_multi + 1 for _, _, n in _ands)
        ands = PriorityQueue()
        hands = {}
        for _and in _ands:
            ands.put_nowait(_and)
            hands[_and[1]] = _and
        since_last_get = 0
        iters = 0
        while not ands.empty() and iters < 500:
            l, ha, na = ands.get_nowait()
            sa = set(na[1:])
            biggest_intersection, h_big = 0, None
            for _, hb, nb in hands.values():
                if hb != ha:
                    inter = set(nb[1:]).intersection(sa)
                    if len(inter) > biggest_intersection:
                        biggest_intersection = len(inter)
                        h_big = hb
            if biggest_intersection > threshold:
                _, _, n_big = hands[h_big]
                inter = set(n_big[1:]).intersection(sa)
                dif = sa.difference(inter)
                # adif = ['a'] + list(dif)
                ainter = ['a'] + list(inter)
                # hdif = self.hash_recursive(adif)
                hinter = self.hash_recursive(ainter,insert=False)
                anew = ['a', hinter] + list(dif)
                hnew = self.hash_recursive(anew,insert=False)
                # if hdif not in hands:
                #     dif_and = -len(adif), hdif, adif
                #     ands.put_nowait(dif_and)
                #     hands[hdif] = dif_and
                if hinter not in hands:
                    inter_and = -len(ainter), hinter, ainter
                    ands.put_nowait(inter_and)
                    hands[hinter] = inter_and
                hands.pop(ha)
                if hnew not in hands:
                    _and_new = -len(anew), hnew, anew
                    hands[hnew] = _and_new
                    ands.put_nowait(_and_new)
                since_last_get = 0
            else:
                since_last_get += 1
            iters += 1
        if ands.empty():
            after_lens = sum((len(n) -1) * h_multi + 1 for _, _, n in hands.values())
        else:
            after_lens = initial_lens
        return initial_lens, after_lens

    def count_and_greedy_effect(self, threshold=3):
        _ands = list(
            (-len(na), ha, na) for ha, na in self.nodes.items() if self.is_and(na)
        )
        return self.count_greedy_effect(_ands,threshold=threshold)

    def count_or_greedy_effect(self, threshold=2, include_prob=True):
        _ors = []
        h_multi = 2 if include_prob else 1
        for ha, na in self.nodes.items():
            if self.is_or(na):
                as_a = ['a'] + [h for _, h in na[1:]]
                _ors.append((len(as_a), self.hash_recursive(as_a,insert=False), as_a))
        return self.count_greedy_effect(_ors, h_multi=h_multi, threshold=threshold)

    def size_after_greedy(self, threshold_and=2, threshold_or=3, include_prob=True):
        lit_size = sum(self.is_literal(n) for n in self.nodes.values())
        and_size = self.count_and_greedy_effect(threshold_and)[1]
        or_size = self.count_or_greedy_effect(threshold_or,include_prob=include_prob)[1]
        return lit_size + and_size + or_size




if __name__ == "__main__":
    """
    Hi! Thank you for reading me.
    """
    """
    This will be our initial Belief state.
    'a' marks AND node, 'o' - OR node.
    Literals are just tuples or lists of (index of variable, value)
    """
    """
    You will see it as a graph so looking deep into is not necessary.
    """
    a = [
        'a',
        [
            'o',
            (0.5, [
                'a',
                [
                    'o',
                    (0.4, [
                        'a',
                        [0, 0],
                        [1, 0]
                    ]),
                    (0.6, [
                        'a',
                        [0, 1],
                        [1, 1]
                    ])
                ],
                [2, 0]
            ]),
            (0.5, [
                'a',
                [0, 0],
                [
                    'o',
                    (0.3, [
                        'a',
                        [1, 0],
                        [2, 0]
                    ]),
                    (0.7, [
                        'a',
                        [1, 1],
                        [2, 2]
                    ])
                ]
            ])

        ],
        [
            'o',
            (0.2, [3, 0]),
            (0.8, [3, 1])
        ]
    ]
    A = AOBS()
    A.root = A.hash_recursive(a)
    # A.draw(window_header="initial")
    """
    To initialize AOBS we recursively hash all subgraphs. 
    hash_recursive() has insert=True by default, so (hash, subgraph) are memorized in A.nodes
    """
    action = \
        ['o',
         (0.1, [
             'a',
             [0, 5],
             [1, 5]
         ]),
         (0.9, [
             'a',
             [0, 7],
             [1, 7]]
          )
         ]
    """
    Now let's just draw the action postconditions:
    """
    B = AOBS()
    B.root = B.hash_recursive(action)
    # B.draw(window_header="action")
    """
    We apply actions to selected substates.
    We select substates by following logical formula: f(v0) and f(v1) and ... and f(vN)
    some f(v_i) can be True(v_i).
    So we pass list  [(v_i, f(v_i)), ...]
    """
    A.act([(2, lambda c: c > 0)], action, debug_draw=True)
    # A.draw(window_header="applied first\nfor c > 0")
    """
    Pfff.. magic happened.
    """
    A.act([(2, lambda c: c > 0), (1, lambda b: b == 7)], action)
    # A.draw(window_header="then for \nc > 0 and b == 7")
    """
    And once more time!
    """
    """
    To look inside turn on the parameter "debug_draw"
    """
    A.act([(0, lambda a: a == 7)], action, debug_draw=False)
    A.act([(0, lambda a: a == 7)], action, debug_draw=False)
    A.act([(0, lambda a: a == 7)], action, debug_draw=False)
    A.act([(0, lambda a: a == 7)], action, debug_draw=False)
    A.act([(0, lambda a: a == 7)], action, debug_draw=False)
    A.act([(0, lambda a: a == 7)], action, debug_draw=False)
    A.act([(0, lambda a: a == 7)], action, debug_draw=False)
    A.draw(window_header="result")
    C = AOBS()
    C.from_collection(A.as_collection())
    C.draw(window_header="states")
    Gtk.main()
