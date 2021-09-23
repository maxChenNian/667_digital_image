import numpy as np
from scipy.special import comb, perm
import itertools

cards = [10, 5, 5, 4, 4, 3]

import itertools

# for i in itertools.combinations(cards, 2):
#     print(str(i) + " ")

num_sets = set()
new_cards = cards[:]
for i, num in enumerate(cards):

    new_cards.remove(num)
    # print(new_cards)
    print("----------------------------------------------------------")
    for num2 in range(len(new_cards)):

        for aa in itertools.combinations(new_cards, num2 + 1):
            print(''.join(str(aa)), end=" ")

        print("\n")

        for aa in itertools.combinations(new_cards, num2 + 1):
            he = np.sum(aa) + num
            num_sets.add(he)
            print(''.join(str(he)), end=" ")

        print("\n")

print()

# num_sets = set()
#
# for num in range(len(cards)):
#
#     for i in itertools.combinations(cards, num + 1):
#         print(''.join(str(i)), end=" ")
#
#     print("\n")
#
#     for i in itertools.combinations(cards, num + 1):
#         he = np.sum(i)
#         num_sets.add(he)
#         print(''.join(str(he)), end=" ")
#
#     print("\n")
#
# print()
#
num_list = list(num_sets)
out_dict = {}
for num_sum in num_list:
    out_dict[str(num_sum)] = []

new_cards = cards[:]
for i, num in enumerate(cards):
    new_cards.remove(num)

    for num2 in range(len(new_cards)):
        for aa in itertools.combinations(new_cards, num2 + 1):

            he = np.sum(aa) + num
            for num_sum in num_list:
                if he == num_sum:
                    bb = list(aa)[:]
                    bb.append(int(num))
                    out_dict[str(he)].append(bb)

print()
