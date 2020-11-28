import torch
import numpy as np
import copy
import random

ENCODING_DICT1={'H':0, 'C':1, 'D':2, 'S':3}
ENCODING_DICT2={'2':0, '3':1, '4':2, '5':3, '6':4, '7':5, '8':6, '9':7, '10':8, 'J':9, 'Q':10, 'K':11, 'A':12}
DECODING_DICT1=['H', 'C', 'D', 'S']
DECODING_DICT2=['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
SCORE_DICT={'SQ':-100,'DJ':100,'C10':0,
            'H2':0,'H3':0,'H4':0,'H5':-10,'H6':-10,'H7':-10,'H8':-10,'H9':-10,'H10':-10,
            'HJ':-20,'HQ':-30,'HK':-40,'HA':-50,'JP':-60,'JG':-70}
TRAINING = True
def card_to_vecpos(card):
    return ENCODING_DICT1[card[0]] * 13 + ENCODING_DICT2[card[1:]]

def vec_to_card(vec):
    pos = np.where(vec == 1)[0][0]
    return DECODING_DICT1[pos//13]+DECODING_DICT2[pos%13]


def pos_to_card(pos):
    return DECODING_DICT1[pos // 13] + DECODING_DICT2[pos % 13]


def card_to_color(card):
    return ENCODING_DICT1[card[0]]

def pos_to_color(pos):
    return DECODING_DICT1[pos//13]

def a_histoire_relatif(histoire, quel_player):
    res = np.zeros((4, 13, 56))
    for i in range(0, len(histoire), 2):
        res[(histoire[i] - quel_player + 4) % 4][i // 8][4 + histoire[i + 1]] = 1
    for i in range(len(histoire) // 8):
        for j in range(4):
            if i * 8 + j * 2 < len(histoire):
                res[(histoire[i * 8 + j * 2] - quel_player + 4) % 4][i][j] = 1
    return res


def a_standard_input(histoire, quel_player, initial_cards):
    if len(initial_cards) < 20:
        initial_cards = initial_to_formatted(initial_cards)
    his = a_histoire_relatif(histoire, quel_player)
    his = torch.flatten(torch.tensor(his), start_dim=0, end_dim=1)
    ini_c = np.zeros((1,56))
    ini_c[0][4:] = initial_cards
    input = torch.cat((his,torch.tensor(ini_c)),0).unsqueeze(0)
    #input[0,0,52,4:] = initial_cards
    return input

def a_standard_input_v(histoire, quel_player, initial_cards):
    if len(initial_cards) < 20:
        initial_cards = initial_to_formatted(initial_cards)
    his = a_histoire_relatif(histoire, quel_player).flatten()
    input = np.concatenate((his, initial_cards))
    return torch.tensor(input)


def initial_to_formatted(initialcards):
    res = np.zeros(52)
    for i in initialcards:
        res[card_to_vecpos(i)] = 1

    return res


def trouver_les_choix_feasibles(initial_vec_r, played_vec_r, color_of_this_turn):
    if len(initial_vec_r) > 20:
        initial_vec = initial_vec_r
    else:
        initial_vec = initial_to_formatted(initial_vec_r)
    if len(played_vec_r) > 51 :#and type(played_vec_r[0]) in {type(1), type(1.)} :
        played_vec = played_vec_r
    else:
        played_vec = initial_to_formatted(played_vec_r)
    # state is already a 1-dim vector
    # returns a np 01 array
    whats_left = initial_vec - played_vec
    empty_color = False
    if color_of_this_turn == 'A':
        empty_color = True
    elif whats_left[card_to_vecpos(color_of_this_turn + '2'):(card_to_vecpos(color_of_this_turn + 'A')+1)].sum() < 1:
        empty_color = True
    if empty_color:
        return whats_left

    pos = np.where(whats_left == 1)[0]
    pos = pos[pos >= card_to_vecpos(color_of_this_turn + '2')]
    pos = pos[pos <= card_to_vecpos(color_of_this_turn + 'A')]

    res = np.zeros(52)
    for i in range(len(pos)):
        res[pos[i]] = 1
    return res

def myhisroty_2_lchistory(history):
    res = []
    for i in range(len(history)//8):
        resi = [history[8*i], pos_to_card(history[8*i+1]),  pos_to_card(history[8*i+3]),  pos_to_card(history[8*i+5]),  pos_to_card(history[8*i+7])]
        res.append(resi)
    return res


def find_scores(history):
    res = [[],[],[],[]]
    for i in range(len(history)//8):
        winner = judge_winner([pos_to_card(history[i*8+1]),pos_to_card(history[i*8+3]),pos_to_card(history[i*8+5]),pos_to_card(history[i*8+7])])
        winner = history[i*8 + winner*2]
        for j in range(4):
            card = pos_to_card(history[i*8+1+j*2])
            if card in SCORE_DICT.keys():
                res[winner].append(card)
    return res


def calc_partial_score(history):
    if len(history) % 8 != 0:
        print('lh', len(history))
        raise Exception('Error: history incomplete')
    score = np.zeros(4)
    has_score_flag = [False, False, False, False]
    c10_flag = [False, False, False, False]
    heart_count = np.zeros(4)
    # calc points
    for turn in range(len(history) // 8):
        gagneur = 0
        for i in [1, 2, 3]:
            if (pos_to_card(history[8 * turn + 2 * i + 1])[0] == pos_to_card(history[8 * turn + 2 * gagneur + 1])[
                0]) & \
                    (ENCODING_DICT2[pos_to_card(history[8 * turn + 2 * i + 1])[1:]] > ENCODING_DICT2[
                        pos_to_card(history[8 * turn + 2 * gagneur + 1])[1:]]):
                gagneur = i
        for people in range(4):
            if history[8 * turn + 2 * gagneur] == people:
                for players in range(4):
                    pos = history[turn * 8 + players * 2 + 1]
                    i = DECODING_DICT1[pos // 13] + DECODING_DICT2[pos % 13]
                    # print(i)
                    if i in SCORE_DICT.keys():
                        if i == "C10":
                            c10_flag[people] = True
                        else:
                            score[people] += SCORE_DICT[i]
                            has_score_flag[people] = True
                        if i.startswith('H') or i.startswith('J'):
                            heart_count[people] += 1
    for people in range(4):
        # check whole Hearts
        if heart_count[people] == 13:
            score[people] += 400
            # settle transformer
        if c10_flag[people] == True:
            if has_score_flag[people] == False:
                score[people] += 50
            else:
                score[people] *= 2

    if TRAINING:
        score[0] = score[0] + score[2]
        score[1] = score[1] + score[3]
        a = score[0]
        b = score[1]
        score[0] = a - b
        score[1] = b - a
        score[2] = score[0]
        score[3] = score[1]
    return score


def whats_left(initial, history, which_player):
    res = copy.deepcopy(initial)
    for i in range(len(history)//2):
        if history[2*i] == which_player:
            card = pos_to_card(history[2*i+1])
            if card in initial:
                res.remove(card)
    return res


def judge_winner(cards_dans_ce_term):
    gagneur = 0
    for i in [1, 2, 3]:
        if (cards_dans_ce_term[i][0] == cards_dans_ce_term[gagneur][0]) & \
                (ENCODING_DICT2[cards_dans_ce_term[i][1:]] > ENCODING_DICT2[cards_dans_ce_term[gagneur][1:]]):
            gagneur = i
    return gagneur


def compter_couleur( card_list):
    res_count = [[[], [], [], []], [[], [], [], []], [[], [], [], []]]
    for i in range(3):
        for j in range(len(card_list[i])):
            res_count[i][ENCODING_DICT1[card_list[i][j][0]]].append(card_list[i][j])
    return res_count


def toAcheck(empty_color, c1, discard_C, potential_C, A_card, B_card):
    # A can receive c1
    a2c_candidate = []
    # print("a card", A_card)
    for acard in A_card:
        if empty_color[3][card_to_color(acard)] == 0:
            a2c_candidate.append(acard)
            break
    if len(a2c_candidate) > 0:
        # 2-rotation can solve the problem
        # print("c1:", c1)
        discard_C.remove(c1)
        potential_C.append(a2c_candidate[0])
        A_card.remove(a2c_candidate[0])
        A_card.append(c1)
        # print("pos 1,", discard_C)
        if len(discard_C) == 0:
            return True
        else:
            return False
            # raise Exception("not returned")
    else:
        # 3-rotation is required
        a2b_candidate = []
        for acard in A_card:
            if empty_color[2][card_to_color(acard)] == 0:
                a2b_candidate.append(acard)
                break
        b2c_candidate = []
        for acard in B_card:
            if empty_color[3][card_to_color(acard)] == 0:
                b2c_candidate.append(acard)
                break
        if (len(a2b_candidate) == 0) or (len(b2c_candidate) == 0):
            return toBcheck(empty_color, c1, discard_C, potential_C, A_card, B_card)
        A_card.remove(a2b_candidate[0])
        A_card.append(c1)
        B_card.remove(b2c_candidate[0])
        B_card.append(a2b_candidate[0])

        discard_C.remove(c1)
        potential_C.append(b2c_candidate[0])
        # print("pos2,",discard_C)
        if len(discard_C) == 0:
            return True
        else:
            return False
        # raise Exception("not returned")


def toBcheck(empty_color, c1, discard_C, potential_C, A_card, B_card):
    # B can receive c1
    b2c_candidate = []
    for acard in B_card:
        if empty_color[3][card_to_color(acard)] == 0:
            b2c_candidate.append(acard)
            break
    if len(b2c_candidate) > 0:
        # 2-rotation can solve the problem
        discard_C.remove(c1)
        potential_C.append(b2c_candidate[0])
        B_card.remove(b2c_candidate[0])
        B_card.append(c1)
        if len(discard_C) == 0:
            return True
        else:
            return False
    else:
        # 3-rotation is required
        a2c_candidate = []
        for acard in A_card:
            if empty_color[3][card_to_color(acard)] == 0:
                a2c_candidate.append(acard)
                break
        b2a_candidate = []
        for acard in B_card:
            if empty_color[1][card_to_color(acard)] == 0:
                b2a_candidate.append(acard)
                break
        if (len(a2c_candidate) == 0) or (len(b2a_candidate) == 0):
            print("A card 3iems:", A_card)
            print("B card 3iems:", B_card)
            print("C card:", potential_C)
            print("C discard:", discard_C)
            print("empty condition:", empty_color)
            raise Exception("Cannot find a 3-rotation!")

        A_card.remove(a2c_candidate[0])
        A_card.append(b2a_candidate[0])
        B_card.remove(b2a_candidate[0])
        B_card.append(c1)

        discard_C.remove(c1)
        potential_C.append(a2c_candidate[0])
        if len(discard_C) == 0:
            return True
        else:
            return False


def ranger_convenable( card_list, empty_color, total_cards):
    # arrange cards for A
    potential_A = []
    remaining_pool = copy.deepcopy(card_list)
    A_card = []
    B_card = []
    # C_card = []
    for acard in card_list:
        if empty_color[1][card_to_color(acard)] == 0:
            potential_A.append(acard)
    random.shuffle(potential_A)

    A_card = potential_A[0:int(total_cards[1])]
    for acard in A_card:
        remaining_pool.remove(acard)

    # arrange cards for B:
    potential_B = []
    discard_B = []
    for acard in remaining_pool:
        if empty_color[2][card_to_color(acard)] == 0:
            potential_B.append(acard)
        else:
            discard_B.append(acard)
    if len(potential_B) < total_cards[2]:
        candidate_from_a = []
        for acard in A_card:
            if empty_color[2][card_to_color(acard)] == 0:
                candidate_from_a.append(acard)
        for onediscard in discard_B:
            if empty_color[1][card_to_color(onediscard)] == 0:
                card_from_A = candidate_from_a[0]
                potential_B.append(card_from_A)
                candidate_from_a.remove(card_from_A)
                A_card.remove(card_from_A)
                A_card.append(onediscard)
                if len(potential_B) >= total_cards[2]:
                    break
    random.shuffle(potential_B)
    B_card = potential_B[0:int(total_cards[2])]

    # arrange cards for C
    remaining_pool = copy.deepcopy(card_list)
    for acard in A_card:
        remaining_pool.remove(acard)
    for acard in B_card:
        remaining_pool.remove(acard)
    potential_C = []
    discard_C = []
    for acard in remaining_pool:
        if empty_color[3][card_to_color(acard)] == 0:
            potential_C.append(acard)
        else:
            discard_C.append(acard)
    while len(discard_C) > 0:
        # further adjustments
        c1 = discard_C[0]
        if empty_color[1][card_to_color(c1)] == 0:
            flag = toAcheck(empty_color, c1, discard_C, potential_C, A_card, B_card)
            # print(flag, discard_C, "to a")
            if flag:
                break
        elif empty_color[2][card_to_color(c1)] == 0:
            flag = toBcheck(empty_color, c1, discard_C, potential_C, A_card, B_card)
            # print(flag, discard_C, 'to B')
            if flag:
                break
        else:
            print("A card 3iems:", A_card)
            print("B card 3iems:", B_card)
            print("C card:", potential_C)
            print("C discard:", discard_C)
            print("empty condition:", empty_color)
            raise Exception("Cannot find a 3-rotation!")
    A_card.sort()
    B_card.sort()
    potential_C.sort()

    res = [A_card, B_card, potential_C]
    # print("res =", res)
    return res

def un_priori_echantillon(empty_color, all_cards_left, turns_left):
    # empty color is a vector of vector [1][2], first 1 means player 1(on the right), second 2 means hcds, 1 for yes
    cards_left = copy.deepcopy(all_cards_left)
    random.shuffle(cards_left)

    res = ranger_convenable(cards_left, empty_color, turns_left)
    return res


def group_de_echantillon(m, empty_color, all_cards_left, turns_left):
    res = []
    one_sample = un_priori_echantillon(empty_color, all_cards_left, turns_left)
    res.append(one_sample)

    continuous_discard = 0

    while True:
        if continuous_discard >= 5:
            # if continuously discarded 5 samples, break
            break
        if len(res) >= m:
            # if there are enough samples in a pool, break
            break
        one_sample = un_priori_echantillon(empty_color, all_cards_left, turns_left)
        # check if a sample is in pool
        not_in_pool = True
        for i in res:
            if one_sample == i:
                not_in_pool = False
                break
        # if not in pool, calculate another example
        if not_in_pool == False:
            continuous_discard += 1
            continue
        else:
            continuous_discard = 0
            res.append(one_sample)
    return res

if __name__ == "__main__":
    r = find_scores([0, 25, 1, 21, 2, 23, 3, 14, 0, 27, 1, 36, 2, 30, 3, 33, 1, 47, 2, 51, 3, 49, 0, 50, 2, 41, 3, 48, 0, 40, 1, 43, 3, 3, 0, 6, 1, 5, 2, 7, 2, 4, 3, 9, 0, 10, 1, 8, 0, 45, 1, 24, 2, 22, 3, 44, 0, 46, 1, 34, 2, 17, 3, 39, 0, 20, 1, 19, 2, 15, 3, 32, 0, 37, 1, 31, 2, 28, 3, 38, 3, 2, 0, 12, 1, 18, 2, 11, 0, 29, 1, 16])
    print(r)