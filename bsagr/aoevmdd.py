import copy
import xdot

from gi.repository import Gtk


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
        return "L" + (str(h) if h >= 0 else "0"+str(abs(h)))

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
                hs += hl + "-> " + self.hname(hc) + (" [ label=\"%."+"%d" % self.precision + "f\"] ;\n") % pc
            for _, hc in n[1:]:
                hs += self.req_dot_str_ver(hc, colors, drawn)
        if self.is_and(n):
            for hc in n[1:]:
                hs += hl + " -> " + self.hname(hc) + ";\n"
            for hc in n[1:]:
                hs += self.req_dot_str_ver(hc, colors, drawn)
        return hs

    def dot_str(self, colors=None, start=None, window_number=None):
        return \
        """digraph %s { 
                rankdir=TD;
                node[shape=circle];
        
        """ % ("aobs") \
        + ("" if window_number is None or window_number < 0 else "AA[label=\"%d\"shape=rectangle, color=blue];\n"%window_number) \
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
            hnode = ['a'] + sorted([self.hash_recursive(n, insert) if isinstance(n, list) else n for n in node[1:]])
        if self.is_or(node):
            hnode = ['o'] + sorted([(p, self.hash_recursive(n, insert)) if isinstance(n, list) else (p, n) for p, n in node[1:]])
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

    def find_min_clusters(self, h,  variables: dict, colors: dict, required_subset: set, min_clusters: dict):
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
            nn += [hc if hc != h else hnew for hc in n[1:] ]
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
                    # print('pidr: ', self.nodes[m])
                    mh = self.isolate(m, colors)
                    mn = self.nodes[mh]
                    # print('isolated: ', mn, [colors[hi] for _, hi in mn[1:]])
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
                # print("mixed: %d, good: %d"%(len(mixed), len(red)))
                if len(mixed) == 1:
                    black_or = blacks[0]
                    red_and.append(reds[0])
                else:
                    for _ in reds:
                        black_or.append((1, ['a']))
                    for i, r in enumerate(reds):
                        red_and.append(r)
                        for j, (_, b) in black_or[1:]:
                            if j == i:
                                b.append(blacks[i])
                            else:
                                b.append(mixed[i])
                black_and.append(black_or)
                for r in red:
                    red_and.append(r)
                    black_and.append(r)
                # print('black_and: ', black_and)
                # print('red_and: ', red_and)
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

    def act_on(self, h, an, variables: set=None):
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
                return p*n[1][0], nn
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
                    return norm_p, self.hash_recursive(['o'] + [(pc/norm_p, hc) for pc, hc in nn[1:]])
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


    def draw(self, colors=None, start=None, window_number=None):
        win = xdot.DotWindow()
        win.set_dotcode(bytes(self.dot_str(colors, start, window_number or len(self.windows)), 'utf-8'))
        self.windows.append(win)

    def act(self, cfs, action, debug_draw=False):
        true_literals, colors = self.get_true_literals_from_conditions(cfs)
        req_subset = set()
        for tl in true_literals:
            req_subset.add(self.nodes[tl][0])
        req_subset.update(self.variablize(self.hash_recursive(action), {}))
        self.colorize(self.root, colors, {})
        if debug_draw: self.draw(colors=colors)
        variables = {}
        self.variablize(self.root, variables)
        min_clusters = {}
        self.find_min_clusters(self.root, variables, colors, req_subset, min_clusters)
        if debug_draw: self.draw(colors=min_clusters)
        clusters = [h for h, v in min_clusters.items() if v]
        new_clusters = [self.normalize(self.isolate(hcl, colors))[1] for hcl in clusters]
        for hcl, hclnew in zip(clusters, new_clusters):
            nhclnew = self.nodes[hclnew]
            self.colorize(hclnew, colors, {})
            if self.is_or(nhclnew):
                nn = ['o'] + [(pc, self.act_on(hc, action) if colors[hc] is True else hc) for pc, hc in nhclnew[1:]]
                self.replace_with(hcl, self.hash_recursive(nn))
            if self.is_and(nhclnew):
                w = self.act_on(hclnew, action)
                self.replace_with(hcl, w)
        if debug_draw: self.draw();
        _, self.root = self.normalize(self.root)
        self.cleanup()

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
action = ['o', (0.1, ['a', [0,5],[1,5]]), (0.9, ['a', [0,7],[1,7]])]
A.act([(2, lambda c: c > 0)], action)
# A.draw()
A.act([(2, lambda c: c > 0), (1, lambda b: b == 7)], action, debug_draw=False)

A.act([(0, lambda a: a == 7)], action, debug_draw=False)
A.draw()
Gtk.main()