import numpy as np
import torch
from model import PNet
from model import VNet
import copy
import random
import time

ENCODING_DICT1 = {'H': 0, 'C': 1, 'D': 2, 'S': 3}
ENCODING_DICT2 = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, '10': 8, 'J': 9, 'Q': 10, 'K': 11,
                  'A': 12}
DECODING_DICT1 = ['H', 'C', 'D', 'S']
DECODING_DICT2 = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']


def card_to_vecpos(card):
    return ENCODING_DICT1[card[0]] * 13 + ENCODING_DICT2[card[1:]]


def vec_to_card(vec):
    # pos=vec.index(1)
    pos = np.where(vec == 1)[0][0]
    return DECODING_DICT1[pos // 13] + DECODING_DICT2[pos % 13]


def pos_to_card(pos):
    return DECODING_DICT1[pos // 13] + DECODING_DICT2[pos % 13]


def card_to_color(card):
    return ENCODING_DICT1[card[0]]


SCORE_DICT = {'SQ': -100, 'DJ': 100, 'C10': 0,
              'H2': 0, 'H3': 0, 'H4': 0, 'H5': -10, 'H6': -10, 'H7': -10, 'H8': -10, 'H9': -10, 'H10': -10,
              'HJ': -20, 'HQ': -30, 'HK': -40, 'HA': -50, 'JP': -60, 'JG': -70}
TRAINING = True


def a_histoire_relatif(histoire, quel_player):
    res = np.zeros((4, 13, 56))
    for i in range(0, len(histoire), 2):
        res[(histoire[i] - quel_player + 4) % 4][i // 8][4 + histoire[i + 1]] = 1
    for i in range(len(histoire) // 8):
        for j in range(4):
            if i * 8 + j * 2 < len(histoire):
                res[(histoire[i * 8 + j * 2] - quel_player + 4) % 4][i][j] = 1
    return res

'''
def a_standard_input(histoire, quel_player, initial_cards):
    if len(initial_cards) < 20:
        initial_cards = initial_to_formatted(initial_cards)
    his = a_histoire_relatif(histoire, quel_player)
    input = np.zeros((1,1,53,56))
    for i in range(13):
        for j in range(4):
            #print('i,j=',i,',',j)
            input[0,0,4*i+j,:] = his[j][i]
    input[0,0,52,4:] = initial_cards
    return torch.tensor(input)
'''

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
    if len(played_vec_r) > 20:
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

def Q_a_P(q_vec):
    res = np.zeros(52)
    max_pos = 0
    max_Q = -10000000
    return res

def print_Q(player, q_vec):
    print(player, 'Q: [', end='')
    for i in range(52):
        if q_vec[i]>-1000:
            print(pos_to_card(i), '%.3g'%q_vec[i],' ', end='')
    print(']')
def print_P(player, p_vec):
    print(player, 'P: [', end='')
    for i in range(52):
        if p_vec[i]> 0.0001:
            print(pos_to_card(i), '%.3g'%p_vec[i],' ', end='')
    print(']')
class Node:
    def __init__(self):
        self.father = None
        # record the history before playing in this node
        self.history = None
        self.position = None
        self.trun_number = 0
        # private information needed for selecting actions
        self.initial_cards = None
        self.sons = []
        self.V = 0
        self.Q = np.zeros(52)-1000000
        self.N = 0
        self.last_in_the_tree = False

        self.expert_recommendation = None
        # self.legal_actions = None
        self.action_probability = None

    def ajouter_fil(self, son):
        self.sons.append(son)

    def inscrire_histoire(self, histoire):
        self.history = histoire

    def compter_V(self, histoire, position, initial_cards, robot):
        input_vec = a_standard_input(histoire, position, initial_cards)
        out = robot(input_vec)[0][52]
        return input_vec, out

    def chercher_ce_fil(self, son):
        if len(self.sons) == 0:
            return None
        for i in self.sons:
            #print("i history:", i.history, "son history:", son.history)
            if i.history == son.history:
                #print(i.history)
                return i
        return None

    def __del__(self):
        for i in range(len(self.sons)):
            j = self.sons.pop()
            del j

    def delete_sub_tree(self):
        for i in range(len(self.sons)):
            j = self.sons.pop()
            del j

from monsieursi2 import IfPlayer
from interfaces2 import misss

class MCTS:
    def __init__(self,pnet):
        self.searching_stack = []
        self.history = []
        self.dependence_on_tree_search = 0.5
        self.epsilon = 0.1
        self.epsilon1 = 0.05
        self.who_wins_each_turn = []
        self.cards_played = [[], [], [], []]
        self.cards_played_original = [[], [], [], []]
        self.who_wins_each_turn_original = []
        # self.empty_color = np.zeros((4, 4))
        self.m = 20
        self.number_of_posterior_samples = 18 #10
        self.number_of_tree_search = 30
        self.tau = 0.5
        self.weight_vec = np.array([0.25, -0.25, 0.25, -0.25])

        self.robot1 = misss(pnet) #IfPlayer()
    def print_history(self, his):
        n = len(his) // 8
        for i in range(n):
            print(i, "th round:", his[i * 8], ":", pos_to_card(his[i * 8 + 1]),
                  his[i * 8 + 2], ":", pos_to_card(his[i * 8 + 3]),
                  his[i * 8 + 4], ":", pos_to_card(his[i * 8 + 5]),
                  his[i * 8 + 6], ":", pos_to_card(his[i * 8 + 7]))
        if n <= 0:
            n = 0
        print(n, "th round:")
        for i in range(8 * n, len(his), 2):
            if i + 1 < len(his):
                print(his[i], ":", pos_to_card(his[i + 1]))

    def whats_next_position(self, this_play_order, this_cards, card):
        if len(this_cards) + 1 < 4:
            return this_play_order[len(this_cards) + 1]
        else:
            all_cards = copy.deepcopy(this_cards)
            all_cards.append(pos_to_card(card))
            #print("lac:", all_cards)
            winner = self.judge_winner(all_cards)
            return winner

    def etendre_et_estimer_Q_pour_la_premier_fois(self, node, history_i, all_inic, position, this_play_order,
                                                  this_cards, prophet, color):
        history = copy.deepcopy(history_i)
        played_vec = np.zeros(52)
        for i in range(0, len(history), 2):
            if history[i] == position:
                played_vec[history[i + 1]] = 1
        legal_choices = trouver_les_choix_feasibles(all_inic[position], played_vec, color)
        pos_lst = np.where(legal_choices == 1)[0]
        history.append(position)
        if len(pos_lst) ==0:
            print("len",len(pos_lst),"ctt", pos_lst)
            print("no more available")
            print("legal choices are:", legal_choices)
            print("initial cards in position ", position, "is:", all_inic[position])
            print("what has been played: ", played_vec)
            self.print_history(history)
        for pos in pos_lst:
            history.append(pos)
            #print("tc", this_cards)
            next_pos = self.whats_next_position(this_play_order, this_cards, pos)
            new_son = Node()
            new_son.position = next_pos
            in_v, new_son.V = new_son.compter_V(history, next_pos, all_inic[next_pos], prophet)
            if node.position % 2==new_son.position %2:
                node.Q[pos] = new_son.V
            else:
                node.Q[pos] = -new_son.V

            new_son.history = copy.deepcopy(history)
            new_son.father = node
            if len(history_i) + 2 >= 104:
                new_son.last_in_the_tree = True
            node.ajouter_fil(new_son)
            history.pop()

    def compter_couleur(self, card_list):
        res_count = [[[], [], [], []], [[], [], [], []], [[], [], [], []]]
        for i in range(3):
            for j in range(len(card_list[i])):
                res_count[i][ENCODING_DICT1[card_list[i][j][0]]].append(card_list[i][j])
        return res_count

    def toAcheck(self, empty_color, c1, discard_C, potential_C, A_card, B_card):
        # A can receive c1
        a2c_candidate = []
        #print("a card", A_card)
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
            #print("pos 1,", discard_C)
            if len(discard_C) == 0:
                return True
            else:
                return False
                #raise Exception("not returned")
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
            if (len(a2b_candidate)==0) or (len(b2c_candidate)==0):
                return self.toBcheck(empty_color, c1, discard_C, potential_C, A_card, B_card)
            A_card.remove(a2b_candidate[0])
            A_card.append(c1)
            B_card.remove(b2c_candidate[0])
            B_card.append(a2b_candidate[0])

            discard_C.remove(c1)
            potential_C.append(b2c_candidate[0])
            #print("pos2,",discard_C)
            if len(discard_C) == 0:
                return True
            else:
                return False
            #raise Exception("not returned")

    def toBcheck(self, empty_color, c1, discard_C, potential_C, A_card, B_card):
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

    def ranger_convenable(self, card_list, empty_color, total_cards):
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
                flag = self.toAcheck(empty_color, c1, discard_C, potential_C, A_card, B_card)
                #print(flag, discard_C, "to a")
                if flag:
                    break
            elif empty_color[2][card_to_color(c1)] == 0:
                flag = self.toBcheck(empty_color, c1, discard_C, potential_C, A_card, B_card)
                #print(flag, discard_C, 'to B')
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

    def un_priori_echantillon(self, empty_color, all_cards_left, turns_left):
        # empty color is a vector of vector [1][2], first 1 means player 1(on the right), second 2 means hcds, 1 for yes
        cards_left = copy.deepcopy(all_cards_left)
        random.shuffle(cards_left)

        res = self.ranger_convenable(cards_left, empty_color, turns_left)
        return res

    def group_de_echantillon(self, empty_color, all_cards_left, turns_left):
        res = []
        one_sample = self.un_priori_echantillon(empty_color, all_cards_left, turns_left)
        res.append(one_sample)

        continuous_discard = 0
        while True:
            if continuous_discard == 5:
                # if continuously discarded 5 samples, break
                break
            if len(res) == self.m:
                # if there are enough samples in a pool, break
                break
            one_sample = self.un_priori_echantillon(empty_color, all_cards_left, turns_left)
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

    def baysian_inference(self, robot, my_initial_cards, my_position, one_sample, certain_history, device):
        # one sample is arranged in relative history
        log_p = 0
        initial_cards = [[], [], [], []]
        initial_cards[my_position] = my_initial_cards
        for i in range(1, 4):
            initial_cards[(my_position + i) % 4] = one_sample[(i - 1 + 4) % 4] + self.cards_played[
                (my_position + i) % 4]
            if len(initial_cards[(my_position + i) % 4] )!=13:
                print("hidden cards:", one_sample[(i - 1 + 4) % 4])
                print("relative position:", i)
                print("played cards:", self.cards_played[
                (my_position + i) % 4])
                raise Exception("error in prior sampling: card number don't add up to 13")

        for i in range(0, len(certain_history), 2):
            the_player = certain_history[i]
            sub_history = certain_history[0:i]
            std_state_input = a_standard_input(sub_history, certain_history[i], initial_cards[the_player])
            if i % 8 > 0:
                card_pos = certain_history[(i // 8) * 8 + 1]
                color = DECODING_DICT1[card_pos // 13]
            else:
                color = 'A'

            device1 = torch.device('cuda')
            net_out = robot.play_one_card(std_state_input, initial_cards[the_player],
                                          self.cards_played[the_player][0:(i // 8)], color, device1)
            #print('net out:', net_out)
            log_p += np.log(0.00001 + net_out[certain_history[i + 1]])
            #log_p +=1
        return log_p, initial_cards

    def compter_likelihood(self, robot, my_initial_cards, my_position, sample_pool, certain_history, device):
        res = []
        initial_pool = []
        for i in range(len(sample_pool)):
            res_i, initial_i = self.baysian_inference(robot, my_initial_cards, my_position, sample_pool[i],
                                                      certain_history, device)
            res.append(res_i)
            initial_pool.append(initial_i)
        return res, initial_pool

    def chercher_et_choisir(self, history_real, ini_card, my_position, cards_already_played, this_turn_till_me,
                            first_play_order, who_wins_each_turn, couleur, robot, prophet, device):

        root_node = Node()
        root_node.history = copy.deepcopy(history_real)
        root_node.initial_cards = copy.deepcopy(ini_card)
        root_node.position = copy.deepcopy(my_position)
        #root_node.expert_recommendation = robot.play_one_card(a_state(history_real, my_position, ini_card),
         #                                                     ini_card, cards_already_played[my_position],
          #                                                    couleur, device)
        root_node.action_probability = copy.deepcopy(root_node.expert_recommendation)
        self.history = copy.deepcopy(history_real)
        self.cards_played = copy.deepcopy(cards_already_played)
        self.cards_played_original = copy.deepcopy(cards_already_played)
        self.who_wins_each_turn_original = copy.deepcopy(who_wins_each_turn)
        self.who_wins_each_turn = copy.deepcopy(who_wins_each_turn)
        # print("capd:",cards_already_played)
        legal_choices = trouver_les_choix_feasibles(initial_to_formatted(ini_card),
                                                         initial_to_formatted(cards_already_played[my_position]),
                                                         couleur)
        all_cards = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'SJ', 'SQ', 'SK', 'SA',
                     'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'HJ', 'HQ', 'HK', 'HA',
                     'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'DJ', 'DQ', 'DK', 'DA',
                     'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'CJ', 'CQ', 'CK', 'CA']
        time_begin = time.time()
        for i in ini_card:
            all_cards.remove(i)
        for i in range(0, len(self.history), 2):
            if self.history[i] != root_node.position:
                pos = self.history[i + 1]
                all_cards.remove(DECODING_DICT1[pos // 13] + DECODING_DICT2[pos % 13])

        # find out if some player lacks certain color from history
        empty_couleur = np.zeros((4, 4), dtype=int)
        L = len(self.history) // 8
        if len(self.history) % 8 == 0:
            L -= 1
        for i in range(L):
            color = self.history[i * 8 + 1] // 13
            for j in range(1, 4):
                if i * 8 + 1 + 2 + j < len(self.history):
                    the_color = self.history[i * 8 + 1 + 2 * j] // 13
                    if the_color != color:
                        empty_couleur[(self.history[i * 8 + 2 * j] - my_position + 4) % 4][color] = 1

        # calculate number of cards in each player's hand
        turns_left = np.zeros(4)
        for i in range(4):
            turns_left[i] = 13 - len(self.history) // 8
        the_number = len(self.history) // 8
        for i in range(0, len(self.history) % 8, 2):
            turns_left[(self.history[the_number * 8 + i] - my_position + 4) % 4] -= 1

        # prior uniform sampling
        sample_pool = self.group_de_echantillon(empty_couleur, all_cards, turns_left)

        # baysien calculation
        try:
            log_likelihood, initial_pool = self.compter_likelihood(robot, root_node.initial_cards, my_position, sample_pool,
                                                               self.history, device)
        except:
            print("error in baysien calculation")
            print("turns left", turns_left)
            raise("Error in prior sampling")


        # initial pool is in absolute order
        # find the largest
        nops = self.number_of_posterior_samples
        if nops > len(log_likelihood):
            nops = len(log_likelihood)
        mle_pos = np.array(log_likelihood).argsort()[-nops:]

        target = np.zeros((len(mle_pos), 52))
        maxed_probs = np.zeros(len(mle_pos))
        v = np.zeros((len(mle_pos), 52))
        #all_Q = np.zeros(52)
        # calculate each possible private information set
        for i in range(len(mle_pos)):
            # under each private information, conduct m Monte Carlo tree searches
            v_vec_i =self.naive_cherche(root_node, initial_pool[mle_pos[i]], this_turn_till_me, first_play_order,
                                      prophet, device)
            v[i] = v_vec_i
            maxed_probs[i] = np.exp(log_likelihood[mle_pos[i]])



        # average max approximation: average the best policy under different private information
        final_v = np.zeros(52)
        maxed_probs = maxed_probs / maxed_probs.sum()

        for i in range(len(mle_pos)):
            #final_v = final_v +  v[i]/(len(mle_pos))
            final_v = final_v + v[i]*maxed_probs[i]


        maxpos=0
        max_V=-100000000

        for i in range(52):
            if final_v[i]>max_V:
                maxpos = i
                max_V = final_v[i]
        if legal_choices.sum()>1:
            final_target = legal_choices*self.epsilon/(legal_choices.sum()-1)
            final_target[maxpos] = 1-self.epsilon
        else:
            final_target = np.zeros(52)
            final_target[maxpos] = 1
        time_end  = time.time()
        #print('ft', final_target, legal_choices.sum())
        if final_target.sum()>1.01:
            print(final_v)
            print(final_target.sum())
            print(final_target)
            raise Exception('Error: target policy problem')
        sample_output = np.random.multinomial(1, final_target, size=1)
        print("After running",len(mle_pos),
              "projections, robot", my_position, "made a choice: ", vec_to_card(sample_output[0]),'final v is:',final_v[np.where(sample_output[0] == 1)[0][0]],
              "Ça coûte", time_end-time_begin, "seconds")

        #print('v is:', v[0][np.where(sample_output[0] == 1)[0][0]])
        #print('final v',final_v)
        #print('which one:', np.where(sample_output[0] == 1)[0][0])
        return vec_to_card(sample_output[0]), a_standard_input(root_node.history, my_position, ini_card), \
               final_target, legal_choices, a_standard_input_v(root_node.history, my_position, ini_card), final_v[np.where(sample_output[0] == 1)[0][0]]

    def naive_cherche(self, root_node, hidden_information, this_turn_till_me, first_play_order, prophet,
                         device):
        self.searching_stack.append(root_node)

        self.play_order = copy.deepcopy(first_play_order)
        root_pos = root_node.position

        color_of_this_turn = 'A'
        if len(this_turn_till_me) > 0:
            color_of_this_turn = this_turn_till_me[0][0]
        card_played_in_this = copy.deepcopy(this_turn_till_me)

        history = copy.deepcopy(root_node.history)
        played_vec = np.zeros(52)
        for i in range(0, len(history), 2):
            if history[i] == root_pos:
                played_vec[history[i + 1]] = 1
        legal_choices = trouver_les_choix_feasibles(hidden_information[root_pos], played_vec, color_of_this_turn)

        pos_lst = np.where(legal_choices == 1)[0]

        v_vec = np.zeros(52)-1000000

        for i in range(len(pos_lst)):
            card_played_in_this.append(pos_to_card(pos_lst[i]))
            history.append(root_pos)
            history.append(pos_lst[i])
            lth = len(card_played_in_this)
            if lth == 1:
                color_of_this_turn = pos_to_card(pos_lst[i])[0]
            for j in range(lth,4):
                card_from_if, _ = self.robot1.play_one_card([[], this_turn_till_me], self.play_order[j], history, hidden_information[self.play_order[j]],
                                              self.cards_played_original[self.play_order[j]], color_of_this_turn, True)
                card_played_in_this.append(card_from_if)
                self.cards_played[self.play_order[j]].append(card_from_if)

                history.append(self.play_order[j])
                history.append(card_to_vecpos(card_from_if))
            #print(history)
            sc = calc_partial_score(history)-calc_partial_score(history[0:(len(history)-8)])
            r_t = sc[root_pos]
            #print('sc',sc)

            winner_order = self.judge_winner(card_played_in_this)
            winner = self.play_order[winner_order]
            next_play_order = [winner, (winner+1)%4, (winner+2)%4,(winner+3)%4]
            #input_vec = a_standard_input_v(history, self.play_order[winner], hidden_information[self.play_order[winner]])
            len_his_ori = len(history)
            color_of_this_turn = 'A'

            if len_his_ori<=48*2:
                value_vec = np.zeros(4)
                for next_rounds in range(4):
                    #value_vec[next_rounds] = prophet.vnet(input_vec)
                    card_from_if, value_vec[next_rounds] = self.robot1.play_one_card([[], this_turn_till_me], next_play_order[next_rounds], history,
                                                             hidden_information[next_play_order[next_rounds]],
                                                             self.cards_played[next_play_order[next_rounds]],
                                                             color_of_this_turn, True)
                    if next_rounds ==0:
                        color_of_this_turn = card_from_if[0]
                    card_played_in_this.append(card_from_if)
                    history.append(self.play_order[next_rounds])
                    history.append(card_to_vecpos(card_from_if))
                    input_vec = a_standard_input_v(history, next_play_order[next_rounds], hidden_information[next_play_order[next_rounds]])
                v_tp1 = sum(value_vec*self.weight_vec)
            else:
                v_tp1 = 0#self.robot1.pret(input_vec)[52]

            if len_his_ori==52*2:
                v_tp1 = 0

            self.cards_played = copy.deepcopy(self.cards_played_original)
            history = copy.deepcopy(root_node.history)
            card_played_in_this = copy.deepcopy(this_turn_till_me)
            #if i%10 ==0:
            #    print('rt=',r_t)
            if self.play_order[winner]%2==root_pos%2:
                v_vec[pos_lst[i]] = r_t + 0.95*v_tp1
            else:
                v_vec[pos_lst[i]] = r_t - 0.95*v_tp1

        return v_vec

    def judge_winner(self, cards_dans_ce_term):
        gagneur = 0
        for i in [1, 2, 3]:
            if (cards_dans_ce_term[i][0] == cards_dans_ce_term[gagneur][0]) & \
                    (ENCODING_DICT2[cards_dans_ce_term[i][1:]] > ENCODING_DICT2[cards_dans_ce_term[gagneur][1:]]):
                gagneur = i
        return gagneur

    def calc_score(self):
        #print("wwet:",self.who_wins_each_turn)
        score = np.zeros(4)
        has_score_flag = [False, False, False, False]
        c10_flag = [False, False, False, False]
        heart_count = np.zeros(4)
        # calc points
        for people in range(4):
            for turn in range(13):
                if self.who_wins_each_turn[turn] == people:
                    for players in range(4):
                        pos = self.history[turn * 8 + players * 2 + 1]
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
            score[0] = a-b
            score[1] = b-a
            score[2] = score[0]
            score[3] = score[1]
        return score

def calc_partial_score(history):
        if len(history)%8!=0:
            print('lh',len(history))
            raise Exception('Error: history incomplete')
        score = np.zeros(4)
        has_score_flag = [False, False, False, False]
        c10_flag = [False, False, False, False]
        heart_count = np.zeros(4)
        # calc points
        for turn in range(len(history) // 8):
            gagneur = 0
            for i in [1, 2, 3]:
                if (pos_to_card(history[8 * turn + 2 * i + 1])[0] == pos_to_card(history[8 * turn + 2 * gagneur + 1])[0]) & \
                        (ENCODING_DICT2[pos_to_card(history[8 * turn + 2 * i + 1])[1:]] > ENCODING_DICT2[
                            pos_to_card(history[8 * turn + 2 * gagneur + 1])[1:]]):
                    gagneur = i
            for people in range(4):
                if history[8*turn+2*gagneur] == people:
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
            score[0] = a-b
            score[1] = b-a
            score[2] = score[0]
            score[3] = score[1]
        return score