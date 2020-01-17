import copy


class BeliefStateAcyclicGraph:
    def __init__(self, literal_definitions, literal_link=''):
        self.literal_definitions = literal_definitions
        self.top = {
            'bottom': None,
            'out'   : [
                {
                    'type': 'literal',
                    'link': literal_link,
                    'prob': 1
                }
            ],
            'in'    : [],
            'type'  : 'node'
        }
        self.top['bottom'] = {
            'in'  : [
                self.top['out'][0]
            ],
            'out' : [],
            'type': 'node'
        }
        self.top['out'][0]['in'] = [self.top]
        self.top['out'][0]['out'] = [self.top['bottom']]

        self._uuid_prefix = 'a'
        self._uuid_number = 0

    def _tensor_product(self, state):
        # assuming they have similar literal_definitions
        res = BeliefStateAcyclicGraph(self.literal_definitions)
        res.top = copy.deepcopy(self.top)
        second_top = copy.deepcopy(state.top)
        res.top['bottom']['out'] = second_top['out']
        res.top['bottom'] = second_top['bottom']
        second_top.pop('bottom')
        return res

    def _split_product(self, other, probs=[1, 1]):
        # assuming they have similar literal_definitions
        if isinstance(probs, list):
            probs = {
                'self' : probs[0],
                'other': probs[1]
            }
        res = BeliefStateAcyclicGraph(self.literal_definitions)
        copies = {
            'self' : copy.deepcopy(self.top),
            'other': copy.deepcopy(other.top)
        }
        for key in ['self', 'other']:
            for l in copies[key]['out']:
                l['prob'] *= probs[key]

        res.top = {
            'out'   : copies['self']['out'] + copies['other']['out'],
            'in'    : [],
            'type'  : 'node',
            'bottom': {
                'in'  : copies['self']['bottom']['in'] + copies['other']['bottom']['in'],
                'out' : [],
                'type': 'node'
            }
        }

        for l in res.top['bottom']['in']:
            l['out'] = [res.top['bottom']]

        return res

    def _reduce(self, obj, reduce_library=None):
        if reduce_library is None:
            reduce_library = {}
        if self._is_pure_tensor(obj):
            return obj
        if self._is_pure_split(obj):
            pass

    @staticmethod
    def _count_literals_in_pure_split(obj):
        obj['sets'] = [BeliefStateAcyclicGraph._get_prob_and_set_of_pure_tensor(l) for l in obj['out']]
        literals_counter = {}
        for i, (_set, _prob) in enumerate(obj['sets']):
            for t in _set:
                if t not in literals_counter:
                    literals_counter[t] = {i}
                else:
                    literals_counter[t].add(i)
        return literals_counter

    def _optimize_pure_split(self, obj, reduce_library):
        pass

    @staticmethod
    def solveIQP(B, C):
        """
        maximize sum_i x_i * B_i,
        w.r.t. C_ij * x_i * x_j = 0
        :param B: (integer numpy 1d vector) vector of subsets power
        :param C: (boolean numpy 2d vector) constraints on validity of our compressed bsagr
        :return: x_i (boolean numpy 1d vector)
        """
        pass

    @staticmethod
    def _find_bottom_of_pure_split(obj):
        node = obj
        while len(node['out']) == 1 and len(node['in']) < 2:
            node = node['out'][0]
        return node

    @staticmethod
    def _get_prob_and_set_of_pure_tensor( obj):
        result = set()
        prob = 1
        node = obj
        while len(node['out']) == 1 and len(node['in']) < 2:
            if node['type'] == 'literal':
                result.add(node['link'])
                prob *= node['prob']
                node = node['out'][0]

        return result, prob

    @staticmethod
    def _is_pure_tensor(obj):
        node = obj
        is_still_pure_tensor = True
        while len(node['out']) > 0:
            if len(node['in']) > 1:
                break
            if len(node['out']) > 1:
                is_still_pure_tensor = False
                break
            node = node['out'][0]
        return is_still_pure_tensor

    @staticmethod
    def _is_pure_split(obj):
        if obj['out'] < 2:
            return False
        is_still_pure_split = True
        for node in obj['out']:
            if not BeliefStateAcyclicGraph._is_pure_tensor(node):
                is_still_pure_split = False
                break
        return is_still_pure_split

    def _get_next_uuid(self):
        self._uuid_number += 1
        return self._uuid_prefix + str(self._uuid_number)

    def _req_dot_view(self, node, literal_def):
        if 'uuid' not in node:
            node['uuid'] = self._get_next_uuid()
        if 'visited' not in node:
            node['visited'] = True
        else:
            return ""

        dot_str = ""

        symbol = ""
        if node['type'] == 'node':
            if len(node['in']) > 1 or len(node['out']) > 1 or len(node['in']) * len(node['out']) == 0:
                symbol = 'x'
            else:
                symbol = 'o'
        elif node['type'] == 'literal':
            if literal_def:
                symbol = self.literal_definitions[node['link']]
            else:
                symbol = node['link']
            if node['prob'] < 1:
                symbol += '\n' + str(node['prob'])

        dot_str += '\t' + node['uuid'] + ' [label="' + symbol + '"'
        if node['type'] == 'node':
            dot_str += ', shape=plaintext'
        dot_str += ']\n'

        # draw arrows
        if node['type'] == 'node':
            for l in node['out']:
                if 'uuid' not in l:
                    l['uuid'] = self._get_next_uuid()
                dot_str += '\t\t' + node['uuid'] + ' -> ' + l['uuid'] + '\n'

            for l in node['in']:
                if 'uuid' not in l:
                    l['uuid'] = self._get_next_uuid()
                dot_str += '\t\t' + l['uuid'] + ' -> ' + node['uuid'] + '\n'

            # if len(node['out']) == 1 and len(node['in']) == 1:
            #     dot_str += '\t{ rank=same ' + node['out'][0]['uuid'] + ' ' + node['uuid'] + ' ' + \
            #                node['in'][0]['uuid'] + ' }\n'

        # go to next nodes
        for l in node['out']:
            dot_str += self._req_dot_view(l, literal_def)

        return dot_str

    def _req_clear_visited(self, node):
        node.pop('visited')
        node.pop('uuid')
        for l in node['out']:
            if 'visited' in l:
                self._req_clear_visited(l)

    def dot_view(self, literal_def=True):
        self._uuid_number = 0
        dot_str = "digraph {\n"

        dot_str += self._req_dot_view(self.top, literal_def)

        dot_str += "\nrankdir=LR\n}\n"

        self._req_clear_visited(self.top)

        return dot_str


from ruamel import yaml

if __name__ == "__main__":
    lit_def = {}
    t1 = BeliefStateAcyclicGraph(lit_def, 'a = 1')
    t2 = BeliefStateAcyclicGraph(lit_def, 'a = 2')
    t3 = t1._split_product(t2, [0.3, 0.7])
    t4 = BeliefStateAcyclicGraph(lit_def, 'b = 3')
    t5 = t3._tensor_product(t4)
    t6 = BeliefStateAcyclicGraph(lit_def, 'b = 5')
    t7 = BeliefStateAcyclicGraph(lit_def, 'a = 3')
    t8 = t6._tensor_product(t7)
    t9 = t8._split_product(t5, [0.4, 0.6])
    print(t9.dot_view(False), file=open('../test.txt', 'w'))
