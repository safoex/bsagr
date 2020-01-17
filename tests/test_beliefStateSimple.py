from unittest import TestCase
from src.bs_interface import BeliefStateSimple as bss

class TestBeliefStateSimple(TestCase):
    def test_physical_state_equals(self):
        a = {
            'a' : 2,
            'b' : 3,
            'c' : 4
        }
        b = {
            'a' : 2,
            'b' : 3,
            'c' : 4
        }
        self.assertTrue(bss._physical_state_equals(a, b))

        c = {
            'a' : 2,
            'b' : 3
        }
        self.assertFalse(bss._physical_state_equals(a, c))

        c['c'] = 5
        self.assertFalse(bss._physical_state_equals(a, c))

        c['c'] = 4
        self.assertTrue(bss._physical_state_equals(a, c))

        c['c'] = {'a' : 2, 'b': 3}
        self.assertFalse(bss._physical_state_equals(a, c))

        c['c'] = [{'c': 4}]
        self.assertFalse(bss._physical_state_equals(a, c))
        
    def test_physical_state_intersects(self):
        a = {
            'a': 2,
            'b': 3,
            'c': 4
        }
        b = {
            'a': 2,
            'b': 3,
            'c': 4
        }
        self.assertTrue(bss._physical_state_intersects(a, b))
        c = {
            'a': 2,
            'b': 3
        }
        self.assertTrue(bss._physical_state_intersects(a, c))
        c = {
            'e': 3,
            'f': 4
        }
        self.assertFalse(bss._physical_state_intersects(a, c))

        c = {
            'a': 5,
            'b': 4,
            'c': 5
        }
        self.assertTrue(bss._physical_state_intersects(a, c))

    def test_normalize(self):
        bs = bss([
            ({'a': 2}, 0.3),
            ({'a': 3}, 0.2)
        ])

        bs.normalize()
        self.assertEqual(0.6, bs.states[0][1])
        self.assertEqual(0.4, bs.states[1][1])

    def test_simplify(self):
        bs = bss([
            ({'a': 2, 'b': 2}, 0.3),
            ({'a': 2, 'b': 3}, 0.2),
            ({'a': 2, 'b': 2}, 0.4),
            ({'a': 2, 'b': 1}, 0.1)
        ])
        bs.simplify()
        self.assertEqual(3, len(bs.states))
        probs = {
            1: 0.1,
            2: 0.7,
            3: 0.2
        }
        for s, p in bs.states:
            self.assertEqual(probs[s['b']], p)

    def test_eq(self):
        bs1 = bss([
            ({'a': 2, 'b': 2}, 0.3),
            ({'a': 2, 'b': 3}, 0.2)
        ])
        bs2 = bss([
            ({'a': 2, 'b': 2}, 0.4),
            ({'a': 2, 'b': 1}, 0.1)
        ])

        self.assertFalse(bs1 == bs2)

        bs1 = bss([
            ({'a': 2, 'b': 2}, 0.3),
            ({'a': 2, 'b': 3}, 0.2)
        ])
        bs2 = bss([
            ({'a': 2, 'b': 2}, 0.3),
            ({'a': 2, 'b': 3}, 0.2)
        ])

        self.assertTrue(bs1 == bs2)

        bs1 = bss([
            ({'a': 2, 'b': 2}, 0.3),
            ({'a': 2, 'b': 3}, 0.2)
        ])
        bs2 = bss([
            ({'a': 2, 'b': 2}, 0.3)
        ])

        self.assertFalse(bs1 == bs2)

    def test_overwrite_ls_by_rs(self):
        bsa = bss([
            ({'a': 2}, 0.4),
            ({'a': 3}, 0.6)
        ])
        bsb = bss([
            ({'b': 2}, 0.5),
            ({'b': 3}, 0.5)
        ])
        bsc = bss([
            ({'a': 2, 'b': 2}, 0.2),
            ({'a': 3, 'b': 2}, 0.3),
            ({'a': 2, 'b': 3}, 0.2),
            ({'a': 3, 'b': 3}, 0.3)
        ])
        self.assertEqual(bsc, bss._overwrite_ls_by_rs(bsa, bsb))
        self.assertEqual(bsc, bss._overwrite_ls_by_rs(bsb, bsa))

    def test_and(self):
        bsa = bss([
            ({'a': 2}, 0.4),
            ({'a': 3}, 0.6)
        ])
        bsb = bss([
            ({'b': 2}, 0.5),
            ({'b': 3}, 0.5)
        ])
        bsc = bss([
            ({'a': 2, 'b': 2}, 0.2),
            ({'a': 3, 'b': 2}, 0.3),
            ({'a': 2, 'b': 3}, 0.2),
            ({'a': 3, 'b': 3}, 0.3)
        ])
        # print(bss._overwrite_ls_by_rs(bsa, bsb).states)
        # print((bsa and bsb).states)
        self.assertEqual(bsc, bsa & bsb)

    def test_or(self):
        bsc = bss([
            ({'a': 2, 'b': 2}, 0.2),
            ({'a': 3, 'b': 2}, 0.3),
            ({'a': 2, 'b': 3}, 0.2),
            ({'a': 3, 'b': 3}, 0.3)
        ])
        bsa = bss([
            ({'a': 3, 'b': 2}, 0.3),
            ({'a': 2, 'b': 3}, 0.2)
        ])
        bsb = bss([
            ({'a': 2, 'b': 2}, 0.2),
            ({'a': 3, 'b': 3}, 0.3)
        ])

        self.assertEqual(bsc, bsa | bsb)

    def test_mul(self):
        bsa = bss([
            ({'a': 2}, 0.4),
            ({'a': 3}, 0.6)
        ])
        bsb = bss([
            ({'b': 2}, 0.5),
            ({'b': 3}, 0.5)
        ])
        bsc = bss([
            ({'a': 2, 'b': 2}, 0.2),
            ({'a': 3, 'b': 2}, 0.3),
            ({'a': 2, 'b': 3}, 0.2),
            ({'a': 3, 'b': 3}, 0.3)
        ])
        self.assertEqual(bsc, bsb * bsa)
        self.assertEqual(bsc, bsa * bsb)

    def test_floordiv(self):
        bsc = bss([
            ({'a': 2, 'b': 2}, 0.2),
            ({'a': 3, 'b': 2}, 0.3),
            ({'a': 2, 'b': 3}, 0.2),
            ({'a': 3, 'b': 3}, 0.3)
        ])
        bsd = bss([
            ({'b': 3}, 0.4)
        ])
        bse = bss([
            ({'a': 2, 'b': 3}, 0.2),
            ({'a': 3, 'b': 3}, 0.3)
        ])
        bsf = bsc / bsd
        self.assertEqual(bse, bsf)

    def test_truediv_and_apply(self):
        bsc = bss([
            ({'a': 2, 'b': 2}, 0.2),
            ({'a': 3, 'b': 2}, 0.3),
            ({'a': 2, 'b': 3}, 0.2),
            ({'a': 3, 'b': 3}, 0.3)
        ])
        bsd = bss([
            ({'b': 3}, 0.4)
        ])
        bse = bss([
            ({'a': 2, 'b': 3}, 0.2),
            ({'a': 3, 'b': 3}, 0.3)
        ])
        bsf = bsc // bsd
        self.assertEqual(bse, bsf)

        bsf.apply({'b': 2})
        bsc.simplify()
        bsg = bss([
            ({'a': 2, 'b': 2}, 0.4),
            ({'a': 3, 'b': 2}, 0.6),
        ])
        self.assertEqual(bsg, bsc)

        bsc.apply({'a': 2})
        bsh = bss([({'a': 2, 'b': 2}, 1)])
        self.assertEqual(bsh, bsc)


