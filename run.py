from lib2to3.pgen2 import token
from typing import Optional
import random
# import rightTime
# from fireWork.fireIn import printHello
# rightTime.fun()
class A():
    def __init__(self, data) -> None:
        self.__data__ = data
# print(A(1).data)

def _get_random_mask_indexes(
    tokens,
    masked_lm_prob=0.15,
    do_whole_word_mask=True,
    max_predictions_per_seq=20,
    special_tokens=[],
):
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token in special_tokens:
            continue
        if (
            do_whole_word_mask and len(cand_indexes) >= 1 and token.startswith("##")
        ) and cand_indexes[-1][-1] == i - 1:
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
    random.shuffle(cand_indexes)
    num_to_predict = min(
        max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob)))
    )
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(covered_indexes) >= num_to_predict:
            break
        if len(covered_indexes) + len(index_set) > num_to_predict or any(
            i in covered_indexes for i in index_set
        ):
            continue
        covered_indexes.update(index_set)
    return covered_indexes

if __name__ == "__main__":
    # print(A(1).__data__)
    # printHello()
    tokens =  ['l','##ov','##e', 'hi']
    max_seq_length = 8
    covered_indexes = _get_random_mask_indexes(
        tokens,
    )
    print(covered_indexes)
    label = [
        tokens[pos] if pos in covered_indexes else -1
        for pos in range(max_seq_length)
    ]
    print(label)
    label_mask = [
        1 if pos in covered_indexes else 0 for pos in range(max_seq_length)
    ]
    print(label_mask)
    
