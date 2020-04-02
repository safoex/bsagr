from matplotlib import pyplot as plt
import pickle
import numpy as np
from bsagr.tests.test_aoevmdd import RandomExploration


file = """
experiment1_1585770469.9192953.nds
"""
# total vars 40
file = """
experiment1_1585772151.3842833.nds
"""
# total vars 30
file = """
experiment1_1585774347.2270706.nds
"""
# total vars 20
file = """
experiment1_1585774649.456698.nds
"""
# total vars 50
file = """
experiment1_1585782840.7763858.nds
"""


with open(file[1:-1], "rb") as f:
    res = pickle.load(f)
    print(res[0])
    x = [np.log10(t['sizes']['total']) for t in res]
    y1 = [np.log10(t['sizes']['aobs']) for t in res]
    y2 = [np.log10(t['sizes']['bdd']) for t in res]
plt.scatter(x,y1,marker='*', label="AOBS")
plt.scatter(x,y2, label='BDD')
plt.xlabel("$(N_{states} * vars)$")
plt.ylabel("$log(|G|)$")
plt.legend()
plt.savefig("../gen_img/50.pdf")
# plt.show()