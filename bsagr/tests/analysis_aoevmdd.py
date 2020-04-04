from matplotlib import pyplot as plt
import pickle, itertools
import numpy as np
from bsagr.tests.test_aoevmdd import RandomExploration

files = []
filens = """
experiment1_1585973173.4550555.nds
"""
file = """
experiment1_1585770469.9192953.nds
"""
# total vars 40
files.append("""
experiment1_1585772151.3842833.nds
""")
# total vars 30
files.append("""
experiment1_1585774347.2270706.nds
""")
# total vars 20
files.append("""
experiment1_1585774649.456698.nds
""")
# total vars 50
files.append("""
experiment1_1585782840.7763858.nds
""")
# dif totals
files.append("""
experiment1_1585922085.320314.nds
""")

# by_vars = {}
# dif_vars_filename = "dif_total_vars.nds"
#
# # for file in files:
# #     with open(file[1:-1], "rb") as f:
# #         res = pickle.load(f)
# #         for r in res:
# #             tv = r['total_vars']
# #             if tv not in by_vars:
# #                 by_vars[tv] = []
# #             by_vars[tv].append(r)
# # with open(dif_vars_filename, 'wb') as f:
# #     pickle.dump(by_vars, file=f)
# with open(dif_vars_filename, 'rb') as f:
#     by_vars = pickle.load(f)
#
# sum_by_vars = {}
# sum_by_vars['aobs'] = {tv: .0 for tv in by_vars}
# sum_by_vars['bdd'] = {tv: .0 for tv in by_vars}
# sum_by_vars['N'] = {tv: .0 for tv in by_vars}
# for tv, results in by_vars.items():
#     for s in ['aobs', 'bdd']:
#         for r in results:
#             sum_by_vars[s][tv] += r['sizes']['total'] / r['sizes'][s]
#     sum_by_vars['N'][tv] += 1
# # x = [np.log10(t['sizes']['total']) for t in res]
# # y1 = [np.log10(t['sizes']['aobs']) for t in res]
# # y2 = [np.log10(t['sizes']['bdd']) for t in res]
# x = list(sum_by_vars['aobs'].keys())
# y = {'aobs': [], 'bdd': []}
# for k, tv in itertools.product(['aobs', 'bdd'], x):
#     y[k].append(sum_by_vars[k][tv] / (sum_by_vars['N'][tv]))
# plt.scatter(x, y['aobs'], marker='*', label="AOBS")
# plt.scatter(x, y['bdd'], label='BDD')
# # plt.xlabel("$(N_{states} * vars)$")
# # plt.ylabel("$log(|G|)$")
# plt.xlabel("vars, $|U|$")
# plt.ylabel("$N_{states} / N_{method}$")
# plt.legend()
# plt.savefig("../gen_img/from_vars.pdf")
# plt.show()

def gen_total_vars_plot():
    dif_vars_filename = "dif_total_vars.nds"
    # with open(dif_vars_filename, 'rb') as f:
    #     by_vars = pickle.load(f)
    #
    # sum_by_vars = {}
    # sum_by_vars['aobs'] = {tv: .0 for tv in by_vars}
    # sum_by_vars['bdd'] = {tv: .0 for tv in by_vars}
    # sum_by_vars['N'] = {tv: .0 for tv in by_vars}
    # for tv, results in by_vars.items():
    #     for s in ['aobs', 'bdd']:
    #         for r in results:
    #             sum_by_vars[s][tv] += r['sizes']['total'] / r['sizes'][s]
    #     sum_by_vars['N'][tv] += 1
    # x = list(sum_by_vars['aobs'].keys())
    # y = {'aobs': [], 'bdd': []}
    # for k, tv in itertools.product(['aobs', 'bdd'], x):
    #     y[k].append(sum_by_vars[k][tv] / (sum_by_vars['N'][tv]))
    with open("xy_" + dif_vars_filename,'rb') as f:
        x,y = pickle.load(f)
    plt.figure(figsize=(4,3))
    plt.scatter(x, y['aobs'], marker='*', label="method=AOBS")
    plt.scatter(x, y['bdd'], label='method=BDD')
    plt.xlabel("vars, $|V|$")
    plt.ylabel("$N_{states} / N_{method}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../gen_img/from_vars.pdf", dpi=150)
    plt.show()

# gen_total_vars_plot()
import copy


def gen_by_N():
    with open(filens[1:-1], 'rb') as f:
        results = pickle.load(f)

    sizes = copy.deepcopy(results[0]['sizes'])
    for k in sizes:
        sizes[k] = 0
    sizes['N'] = 0

    by_sizes = {}
    av_by_sizes = {}

    for r in results:
        if r['steps'] not in by_sizes:
            by_sizes[r['steps']] = copy.deepcopy(sizes)
            av_by_sizes[r['steps']] = copy.deepcopy(sizes)

    sizes.pop('N')
    for r in results:
        for s in sizes:
            by_sizes[r['steps']][s] += r['sizes'][s]
        by_sizes[r['steps']]['N'] += 1

    for steps in by_sizes:
        for s in sizes:
            av_by_sizes[steps][s] = by_sizes[steps][s] / by_sizes[steps]['N']

    x = [steps for steps in by_sizes]
    y = {s: [by_sizes[steps][s] for steps in x] for s in sizes}

    plt.figure(figsize=(10, 5))
    plt.scatter(x, np.log(y['aobs']), marker='*', label="method=AOBS")
    plt.scatter(x, np.log(y['bdd']), label='method=BDD')
    plt.scatter(x, np.log(y['aobs_greedy']), marker='*', label="method=AOBS (+greedy)")
    plt.xlabel("steps")
    plt.ylabel("$N_{states} / N_{method}$")
    plt.legend()
    plt.tight_layout()
    # plt.savefig("../gen_img/from_steps.pdf", dpi=150)
    plt.show()

gen_by_N()

