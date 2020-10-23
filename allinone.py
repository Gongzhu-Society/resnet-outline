import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pygame
import random
import time

ENCODING_DICT1={'H':0, 'C':1, 'D':2, 'S':3}
ENCODING_DICT2={'2':0, '3':1, '4':2, '5':3, '6':4, '7':5, '8':6, '9':7, '10':8, 'J':9, 'Q':10, 'K':11, 'A':12}
DECODING_DICT1=['H', 'C', 'D', 'S']
DECODING_DICT2=['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

class PNet(nn.Module):

    def __init__(self):
        #define a model
        super(PNet, self).__init__()
        self.fc1 = nn.Linear(2964, 1482)
        self.fc2 = nn.Linear(1482, 741)
        self.fc3 = nn.Linear(741, 370)
        self.fc4 = nn.Linear(370, 370)

        self.fc41 = nn.Linear(370, 370)
        self.fc42 = nn.Linear(370, 370)
        self.fc43 = nn.Linear(370, 370)
        self.fc44 = nn.Linear(370, 370)

        self.fc5 = nn.Linear(370, 185)
        self.fc6 = nn.Linear(185, 185)
        self.fc7 = nn.Linear(185, 100)
        self.fc8 = nn.Linear(100, 52)



    def forward(self, x):
        x = F.relu(self.fc1(x.float()))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x1 = F.relu(self.fc4(x))
        x = F.relu(self.fc41(x))
        x = F.relu(self.fc42(x))
        x = F.relu(self.fc43(x))
        x = F.relu(self.fc44(x))

        x = F.relu(self.fc5(x+x1))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        return x

class Robot():
    def __init__(self, pnet):
        self.pnet = pnet
        self.beta = 1

    def loss_func_single(self, features, output, legal_choix):
        #print("hello here")
        alpha = 0.001
        bc = 0.0000001
        expd = torch.exp(self.beta * features)
        expd = torch.mul(expd, legal_choix)

        prob = expd / torch.sum(expd)
        similarity = -torch.sum(torch.mul(torch.log(prob+0.000001), output))
        entropy = torch.sum(torch.mul(prob, torch.log(prob+0.0000001)))


        return alpha * similarity + bc * entropy

    def loss_func(self, features_v, output_v, legal_choix_v):
        res = torch.zeros(len(output_v))
        for i in range(0, len(output_v)):
            features = features_v[i]
            output = output_v[i]
            legal_choix = legal_choix_v[i]
            #print("bonjour")
            res[i] = self.loss_func_single(features, output, legal_choix)
            #print("bonjour2")
        return torch.sum(res)/(len(output_v))

    def output_to_probability(self, out_vec, legal_choix):
        expd = torch.exp(self.beta * out_vec)
        #legal_choix_r.to(device)
        expd = torch.mul(expd, legal_choix)
        prob = expd / torch.sum(expd)
        return prob

    def initial_to_formatted(self, initialcards):
        res = np.zeros(52)
        for i in initialcards:
            res[card_to_vecpos(i)] = 1
        return res

    def cards_left(self, initial_vec, played_vec, color_of_this_turn):
        whats_left = initial_vec - played_vec
        #print("whata left are", vec_to_cards(whats_left))
        empty_color = False
        if color_of_this_turn == 'A':
            empty_color = True

        elif whats_left[card_to_vecpos(color_of_this_turn+'2'):(card_to_vecpos(color_of_this_turn+'A')+1)].sum() < 1:
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

    def play_one_card(self, state, initial_cards, cards_played, couleur, device):
        legal_choices = self.cards_left(self.initial_to_formatted(initial_cards), self.initial_to_formatted(cards_played), couleur)

        input = torch.tensor(state)
        #input = torch.tensor(input)
        n = legal_choices.sum()
        # if there is only one choice, we don't need the calculation of q
        if n < 1.5:
            return legal_choices
        # elsewise, we need the policy network
        net_output = self.pnet(input)

        probability = self.output_to_probability(net_output, torch.tensor(legal_choices).to(device))
        if device == 'cpu':
            return probability.detach().numpy()

        prb = probability.detach().cpu().numpy()

        #sample_output = np.random.multinomial(1, prb, size=1)

        return prb

class Memory(torch.utils.data.Dataset):
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self.input_samples = []
        self.target_policy = []
        self.target_value = []
        self.legal_choices = []

        self.first_empty_i = 0
        self.first_empty_tp = 0
        self.first_empty_tv = 0
        self.first_empty_lc = 0

    def __getitem__(self, index):
        return self.input_samples[index], self.target_policy[index], self.target_value[index], self.legal_choices[index]

    def __len__(self):
        return self._max_memory

    def add_input_sample(self, input_sample):
        self.input_samples.append(input_sample)
        self.first_empty_i += 1
        if self.first_empty_i == self._max_memory:
            self.first_empty_i -= 1

    def add_target_policy(self, tgt_policy):
        self.target_policy.append(tgt_policy)
        self.first_empty_tp += 1
        if self.first_empty_tp == self._max_memory:
            self.first_empty_tp -= 1

    def add_target_value(self, tgt_value):
        self.target_value.append(tgt_value)
        self.first_empty_tv += 1
        if self.first_empty_tv == self._max_memory:
            self.first_empty_tv -= 1

    def add_lc_sample(self, lc):
        self.legal_choices.append(lc)
        self.first_empty_lc +=1
        if self.first_empty_lc == self._max_memory:
            self.first_empty_lc -=1

    def clear(self):
        self.first_empty_i = 0
        self.first_empty_tv = 0
        self.first_empty_tp = 0
        self.first_empty_lc = 0
        length = len(self.target_value)
        for i in range(length):
            self.input_samples.pop()
            self.target_policy.pop()
            self.target_value.pop()
            self.legal_choices.pop()

class VNet(nn.Module):
    def __init__(self):
        #define a model
        super(VNet, self).__init__()
        self.fc1 = nn.Linear(2964, 1482)
        self.fc2 = nn.Linear(1482, 741)
        self.fc3 = nn.Linear(741, 370)
        self.fc4 = nn.Linear(370, 185)
        self.fc5 = nn.Linear(185, 100)
        self.fc6 = nn.Linear(100, 50)
        self.fc7 = nn.Linear(50, 20)
        self.fc8 = nn.Linear(20, 1)


    def forward(self, x):
        x = F.relu(self.fc1(x.float()))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        return x


class Prophet():
    def __init__(self, vnet):
        self.vnet = vnet

    def loss_func(self, output_v, target_v):
        res = torch.zeros(len(output_v))
        for i in range(0, len(output_v)):
            res[i] = (output_v[i]-target_v[i])*(output_v[i]-target_v[i])
        return torch.sum(res)/(len(target_v))



import copy
import random





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
    return res.flatten()


def a_standard_input(histoire, quel_player, initial_cards):
    if len(initial_cards) < 20:
        initial_cards = initial_to_formatted(initial_cards)
    his = a_histoire_relatif(histoire, quel_player)
    input = np.concatenate((his, initial_cards))
    # input = np.concatenate((input, legal_choix))
    return torch.tensor(input)


def initial_to_formatted(initialcards):
    res = np.zeros(52)
    for i in initialcards:
        res[card_to_vecpos(i)] = 1
    return res


def a_state(histoire, quel_joueur, ini_c):
    his = a_histoire_relatif(histoire, quel_joueur)
    if len(ini_c) > 20:
        return np.concatenate((his, ini_c))
    return np.concatenate((his, initial_to_formatted(ini_c)))


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

    def compter_V(self, histoire, position, initial_cards, prophet):
        input_vec = a_standard_input(histoire, position, initial_cards)
        out = prophet.vnet(input_vec)
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


ORDER_DICT1 = {'S': -300, 'H': -200, 'D': -100, 'C': 0, 'J': -200}
ORDER_DICT2 = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '1': 10, 'J': 11, 'Q': 12, 'K': 13,
               'A': 14, 'P': 15, 'G': 16}


def cards_order(card):
    return ORDER_DICT1[card[0]] + ORDER_DICT2[card[1]]


def get_nonempty_min(l):
    if len(l) != 0:
        return len(l)
    else:
        return 100


class IfPlayer:
    def cards_left(self, initial_vec, played_vec, color_of_this_turn):
        whats_left = initial_vec - played_vec
        empty_color = False
        if color_of_this_turn == 'A':
            empty_color = True
        elif whats_left[
             card_to_vecpos(color_of_this_turn + '2'):(card_to_vecpos(color_of_this_turn + 'A') + 1)].sum() < 1:
            empty_color = True
        if empty_color:
            return whats_left

        pos = np.where(whats_left == 1)[0]
        pos = pos[pos >= card_to_vecpos(color_of_this_turn + '2')]
        pos = pos[pos <= card_to_vecpos(color_of_this_turn + 'A')]

        # print(pos)
        res = np.zeros(52)
        for i in range(len(pos)):
            res[pos[i]] = 1
        return res

    def initial_to_formatted(self, initialcards):
        res = np.zeros(52)
        for i in initialcards:
            res[card_to_vecpos(i)] = 1
        return res

    def pick_a_card(self, suit, cards_dict, cards_list, cards_on_table):

        try:
            assert len(cards_list) == sum([len(cards_dict[k]) for k in cards_dict])
        except:
            log("", l=3)
        # log("%s, %s, %s, %s"%(self.name,suit,self.cards_on_table,cards_list))
        # 如果随便出
        if suit == "A":
            list_temp = [cards_dict[k] for k in cards_dict]
            list_temp.sort(key=get_nonempty_min)
            # log(list_temp)
            for i in range(4):
                if len(list_temp[i]) == 0:
                    continue
                suit_temp = list_temp[i][0][0]
                # log("thinking %s"%(suit_temp))
                if suit_temp == "S" and ("SQ" not in cards_list) \
                        and ("SK" not in cards_list) and ("SA" not in cards_list):
                    choice = cards_dict["S"][-1]
                    cards_list.remove(choice)
                    cards_dict[choice[0]].remove(choice)
                    return choice
                if suit_temp == "H" and ("HQ" not in cards_list) \
                        and ("HK" not in cards_list) and ("HA" not in cards_list):
                    choice = cards_dict["H"][-1]
                    cards_list.remove(choice)
                    cards_dict[choice[0]].remove(choice)
                    return choice
                if suit_temp == "C" and ("C10" not in cards_list) \
                        and ("CJ" not in cards_list) and ("CQ" not in cards_list) \
                        and ("CK" not in cards_list) and ("CA" not in cards_list):
                    choice = cards_dict["C"][-1]
                    cards_list.remove(choice)
                    cards_dict[choice[0]].remove(choice)
                    return choice
                if suit_temp == "D" and ("DJ" not in cards_list):
                    choice = cards_dict["D"][-1]
                    cards_list.remove(choice)
                    cards_dict[choice[0]].remove(choice)
                    return choice
            for i in range(5):
                choice = random.choice(cards_list)
                if choice not in ("SQ", "SK", "SA", "HA", "HK", "C10", "CJ", "CQ", "CK", "CA", "DJ"):
                    cards_list.remove(choice)
                    cards_dict[choice[0]].remove(choice)
                    return choice
        # 如果是贴牌
        elif len(cards_dict[suit]) == 0:
            for i in (
            "SQ", "HA", "SA", "SK", "HK", "C10", "CA", "HQ", "HJ", "CK", "CQ", "CJ", "H10", "H9", "H8", "H7", "H6",
            "H5"):
                if i in cards_list:
                    choice = i
                    cards_list.remove(choice)
                    cards_dict[choice[0]].remove(choice)
                    return choice
            list_temp = [cards_dict[k] for k in cards_dict]
            list_temp.sort(key=get_nonempty_min)
            for i in range(4):
                if len(list_temp[i]) == 0:
                    continue
                suit_temp = list_temp[i][0][0]
                choice = cards_dict[suit_temp][-1]
                cards_list.remove(choice)
                cards_dict[choice[0]].remove(choice)
                return choice
        # 如果只有这一张
        elif len(cards_dict[suit]) == 1:
            choice = cards_dict[suit][-1]
            cards_list.remove(choice)
            cards_dict[choice[0]].remove(choice)
            return choice

        # 如果是猪并且剩好几张猪牌
        if suit == "S":
            if ("SQ" in cards_list) and (("SK" in cards_on_table) or ("SA" in cards_on_table)):
                choice = "SQ"
                cards_list.remove(choice)
                cards_dict[choice[0]].remove(choice)
                return choice
            if len(cards_on_table) == 4 and ("SQ" not in cards_on_table):
                choice = cards_dict["S"][-1]
                if choice == "SQ":
                    choice = cards_dict["S"][-2]
                cards_list.remove(choice)
                cards_dict[choice[0]].remove(choice)
                return choice
            else:
                if "SA" in cards_on_table[1:]:
                    max_pig = cards_order("SA")
                elif "SK" in cards_on_table[1:]:
                    max_pig = cards_order("SK")
                else:
                    max_pig = cards_order("SQ")
                for i in cards_dict["S"][::-1]:
                    if cards_order(i) < max_pig:
                        choice = i
                        cards_list.remove(choice)
                        cards_dict[choice[0]].remove(choice)
                        return choice
                else:
                    choice = cards_dict["S"][-1]
                    cards_list.remove(choice)
                    cards_dict[choice[0]].remove(choice)
                    return choice
        # 如果是变压器并且草花剩两张以上
        if suit == "C":
            if ("C10" in cards_list) \
                    and (("CJ" in cards_on_table) or ("CQ" in cards_on_table) or \
                         ("CK" in cards_on_table) or ("CA" in cards_on_table)):
                choice = "C10"
                cards_list.remove(choice)
                cards_dict[choice[0]].remove(choice)
                return choice
            if len(cards_on_table) == 4 and ("C10" not in cards_on_table):
                choice = cards_dict["C"][-1]
                if choice == "C10":
                    choice = cards_dict["C"][-2]
                cards_list.remove(choice)
                cards_dict[choice[0]].remove(choice)
                return choice
            else:
                if "CA" in cards_on_table[1:]:
                    max_club = cards_order("CA")
                elif "CK" in cards_on_table[1:]:
                    max_club = cards_order("CK")
                elif "CQ" in cards_on_table[1:]:
                    max_club = cards_order("CQ")
                elif "CJ" in cards_on_table[1:]:
                    max_club = cards_order("CJ")
                else:
                    max_club = cards_order("C10")
                for i in cards_dict["C"][::-1]:
                    if cards_order(i) < max_club:
                        choice = i
                        cards_list.remove(choice)
                        cards_dict[choice[0]].remove(choice)
                        return choice
                else:
                    choice = cards_dict["C"][-1]
                    cards_list.remove(choice)
                    cards_dict[choice[0]].remove(choice)
                    return choice
        # 如果是羊并且剩两张以上
        if suit == "D":
            if len(cards_on_table) == 4 and ("DJ" in cards_dict["D"]) \
                    and ("DA" not in cards_on_table) and ("DK" not in cards_on_table) \
                    and ("DQ" not in cards_on_table):
                choice = "DJ"
                cards_list.remove(choice)
                cards_dict[choice[0]].remove(choice)
                return choice
            choice = cards_dict["D"][-1]
            if choice == "DJ":
                choice = cards_dict["D"][-2]
            cards_list.remove(choice)
            cards_dict[choice[0]].remove(choice)
            return choice
        # 如果是红桃
        if suit == "H":
            max_heart = -1000
            for i in cards_on_table[1:]:
                if i[0] == "H" and cards_order(i) > max_heart:
                    max_heart = cards_order(i)
            for i in cards_dict["H"][::-1]:
                if cards_order(i) < max_heart:
                    choice = i
                    cards_list.remove(choice)
                    cards_dict[choice[0]].remove(choice)
                    return choice
        # log("cannot be decided by rules")
        return 'RIEN'

    def play_one_card(self, info, my_position, history_real, initial_cards, cards_played, couleur, training):
        legal_choices = self.cards_left(self.initial_to_formatted(initial_cards),
                                        self.initial_to_formatted(cards_played), couleur)
        which_turn = info[0]
        cards_in_this_turn = info[1]
        cards_left = copy.deepcopy(initial_cards)
        for i in cards_played:
            cards_left.remove(i)
        cards_dict = {"S": [], "H": [], "D": [], "C": []}
        for i in cards_left:
            cards_dict[i[0]].append(i)
        res = self.pick_a_card(couleur, cards_dict, cards_left, cards_in_this_turn)
        if res != 'RIEN':
            return res

        prb = legal_choices / np.sum(legal_choices)
        sample_output = np.random.multinomial(1, prb, size=1)
        the_card = vec_to_card(sample_output[0])
        return the_card


class MCTS:
    def __init__(self):
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
        self.m = 10
        self.number_of_posterior_samples = 5 #10
        self.number_of_tree_search = 30
        self.tau = 0.5

        self.robot1 = IfPlayer()
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

    def naive_etendre_et_estimer_Q_pour_la_premier_fois(self, node, history_i, all_inic, position, this_play_order,
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
            std_state_input = a_state(sub_history, certain_history[i], initial_cards[the_player])
            if i % 8 > 0:
                card_pos = certain_history[(i // 8) * 8 + 1]
                color = DECODING_DICT1[card_pos // 13]
            else:
                color = 'A'
            net_out = robot.play_one_card(std_state_input, initial_cards[the_player],
                                          self.cards_played[the_player][0:(i // 8)], color, device)
            log_p += np.log(0.00001 + net_out[certain_history[i + 1]])
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
            final_v = final_v + maxed_probs[i] * v[i]


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
        print("After running",len(mle_pos),"*",self.number_of_tree_search,
              "projections, robot", my_position, "made a choice: ", vec_to_card(sample_output[0]),'final v is:',final_v[np.where(sample_output[0] == 1)[0][0]],
              "Ça coûte", time_end-time_begin, "seconds")

        #print('v is:', v[0][np.where(sample_output[0] == 1)[0][0]])
        #print('final v',final_v)
        #print('which one:', np.where(sample_output[0] == 1)[0][0])
        return vec_to_card(sample_output[0]), a_standard_input(root_node.history, my_position, ini_card), \
               final_target, legal_choices, final_v[np.where(sample_output[0] == 1)[0][0]]

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
                card_from_if = self.robot1.play_one_card([[], this_turn_till_me], self.play_order[j], history, hidden_information[self.play_order[j]],
                                              self.cards_played_original[self.play_order[j]], color_of_this_turn, True)
                card_played_in_this.append(card_from_if)
                history.append(self.play_order[j])
                history.append(card_to_vecpos(card_from_if))
            #print(history)
            sc = calc_partial_score(history)-calc_partial_score(history[0:(len(history)-8)])
            r_t = sc[root_pos]
            #print('sc',sc)
            winner_order = self.judge_winner(card_played_in_this)
            winner = self.play_order[winner_order]
            input_vec = a_standard_input(history, self.play_order[winner], hidden_information[self.play_order[winner]])
            v_tp1 = prophet.vnet(input_vec)

            if len(history)==52*2:

                v_tp1 = 0

            history = copy.deepcopy(root_node.history)
            card_played_in_this = copy.deepcopy(this_turn_till_me)
            #if i%10 ==0:
            #    print('rt=',r_t)
            if self.play_order[winner]%2==root_pos%2:
                v_vec[pos_lst[i]] = r_t + 0.99*v_tp1
            else:
                v_vec[pos_lst[i]] = r_t - 0.99*v_tp1

        return v_vec


    def nouvelle_cherche(self, root_node, hidden_information, this_turn_till_me, first_play_order, t, robot, prophet, device):
        self.searching_stack.append(root_node)
        node_considering = root_node
        self.play_order = copy.deepcopy(first_play_order)

        color_of_this_turn = 'A'
        for i in range(len(root_node.history) // 8, 13):
            if i == len(root_node.history) // 8:
                n = len(this_turn_till_me)
                card_played_in_this_turn = copy.deepcopy(this_turn_till_me)
                if len(this_turn_till_me) > 0:
                    color_of_this_turn = this_turn_till_me[0][0]
            else:
                n = 0
                card_played_in_this_turn = []
            for j in range(n, 4):
                # select an action
                node_considering.initial_cards = hidden_information[self.play_order[j]]
                if node_considering.N == 0:
                    # during the first visit, add sons set up prior probability and prior V of sons
                    inic = hidden_information[self.play_order[j]]
                    node_considering.expert_recommendation = robot.play_one_card(
                        a_state(self.history, self.play_order[j], inic),
                        inic, self.cards_played[self.play_order[j]],
                        color_of_this_turn, device)
                    # add sons and calculate prior V value of sons, calculate Q value
                    self.etendre_et_estimer_Q_pour_la_premier_fois(node_considering, self.history, hidden_information,
                                                                   self.play_order[j], self.play_order,
                                                                   card_played_in_this_turn,
                                                                   prophet, color_of_this_turn)
                    actions = np.zeros(52)
                    N = len(node_considering.sons)
                    max_Q = -10000000
                    max_pos = 0
                    if N > 0:
                        for node_j in node_considering.sons:
                            vec_pos = node_j.history[-1]
                            actions[vec_pos] = self.epsilon / N
                            if node_considering.Q[vec_pos] > max_Q:
                                max_pos = vec_pos
                                max_Q = node_considering.Q[vec_pos]
                        if N == 1:
                            # the only son
                            actions[max_pos] = 1
                        elif N > 1:
                            # multiple sons
                            actions[max_pos] = 1 - self.epsilon*(N-1)/N

                    # if there are no sons left, (or the son is a leaf node), leave actions to zero vectors, expert recommendation will do all job
                    node_considering.action_probability = copy.deepcopy(actions) * (self.dependence_on_tree_search) + \
                                                          (1 - self.dependence_on_tree_search) * node_considering.expert_recommendation / (
                                                                  node_considering.N + 1)

                #print("action prob:",node_considering.action_probability)
                the_action = np.where(node_considering.action_probability == np.max(node_considering.action_probability))[0][0]
                #node_considering.position = self.play_order[j]
                card = DECODING_DICT1[the_action // 13] + DECODING_DICT2[the_action % 13]
                if j == 0:
                    color_of_this_turn = card[0]
                # print("initial card of player ",self.play_order[j],"is ", hidden_information[self.play_order[j]])
                # print("he played:", self.cards_played[self.play_order[j]])
                # print("he plays:", card)
                card_played_in_this_turn.append(card)
                self.cards_played[self.play_order[j]].append(card)

                # add the action to history
                self.history.append(self.play_order[j])
                self.history.append(the_action)

                # consider the new node
                new_son = Node()
                new_son.father = self.searching_stack[-1]
                new_son.history = copy.deepcopy(self.history)

                # when first visiting new son, add it as the son of his father, and calculate expert recommendation
                possible_son = new_son.father.chercher_ce_fil(new_son)
                #print("hello")
                if possible_son == None:
                    print("the action:", the_action)
                    print("initial cards:", hidden_information[self.play_order[j]])
                    print("card played for player j:",self.cards_played[self.play_order[j]])
                    print("action probability:", node_considering.action_probability)
                    print("expert recommendation:", node_considering.expert_recommendation)
                    #print("tree search probability:", actions)
                    print("Q:", node_considering.Q)
                    print("nsfh:",new_son.father.history)
                    print("nchis", node_considering.history)
                    print("ncson len:", len(node_considering.sons), "nc.N", node_considering.N)
                    print("new son his:",new_son.history)
                    #print("possible son his:", possible_son.history)
                    for i in node_considering.sons:
                        print("ncs:",i.history)

                    raise Exception("Cannot find this son!")
                else:
                    del new_son
                    self.searching_stack.append(possible_son)
                    node_considering = self.searching_stack[-1]
            # update playorder after each turn
            winner = self.play_order[self.judge_winner(card_played_in_this_turn)]
            # print(winner)
            self.who_wins_each_turn.append(winner)
            self.play_order = [winner, (winner + 1) % 4, (winner + 2) % 4, (winner + 3) % 4]
            color_of_this_turn = 'A'

        # update nodes in the stack
        # the last element in the searching stack is redundant??
        self.searching_stack.pop()

        scores = self.calc_score()
        for i in range(len(self.searching_stack)):
            node_i = self.searching_stack[i]
            node_i.V += scores[node_i.position]
            node_i.N += 1
            if i != 0: # do not have to update root node
                if node_i.father.position % 2 == node_i.position % 2:
                    node_i.father.Q[node_i.history[-1]] = node_i.V / (node_i.N + 1)
                else:
                    node_i.father.Q[node_i.history[-1]] = - node_i.V / (node_i.N + 1)
        for i in range(len(self.searching_stack)):
            node_i = self.searching_stack[i]
            # update the action probability
            # first, find out the greatest value function among sons
            actions = np.zeros(52)
            actions = actions #- 10000000
            max_Q = -100000
            max_pos = 0
            # then use an epsilon greedy strategy for the sons searched
            N = len(node_i.sons)
            if N > 0:
                #print("niq", node_i.Q)
                for node_j in node_i.sons:
                    vec_pos = node_j.history[-1]
                    actions[vec_pos] = self.epsilon / N
                    #print("nj pos", vec_pos)
                    if node_i.Q[vec_pos] > max_Q:
                        max_pos = vec_pos
                        max_Q = node_i.Q[vec_pos]
                if N == 1:
                    # the only son
                    actions[max_pos] = 1
                elif N > 1:
                    # multiple sons
                    actions[max_pos] = 1 - self.epsilon * (N-1) / N

            # if there are no sons left, (or the son is a leaf node), leave actions to zero vectors, expert recommendation will do all job
            node_i.action_probability = actions * (self.dependence_on_tree_search) + \
                                        (1 - self.dependence_on_tree_search) * node_i.expert_recommendation / (
                                                    node_i.N + 1)
        # at the end of each mcts, we need to pop stack, and delete history
        self.history = copy.deepcopy(root_node.history)
        self.cards_played = copy.deepcopy(self.cards_played_original)
        self.who_wins_each_turn = copy.deepcopy(self.who_wins_each_turn_original)

        for i in range(len(self.searching_stack)):
            self.searching_stack.pop()

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

class GameRunner:
    def __init__(self):
        self.initial_cards = [[]]
        # history from judge's view
        self.history = []
        # another quick save for history
        self.cards_sur_table = [[]]

        self.play_order = [0, 1, 2, 3]
        # 13 elements, representing who wins each turn
        self.who_wins_each_turn = []

    def new_shuffle(self):
        cards = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'SJ', 'SQ', 'SK', 'SA',
                 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'HJ', 'HQ', 'HK', 'HA',
                 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'DJ', 'DQ', 'DK', 'DA',
                 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'CJ', 'CQ', 'CK', 'CA']
        random.shuffle(cards)
        self.play_order = [0, 1, 2, 3]
        random.shuffle(self.play_order)
        for i in [1,2,3]:
            self.play_order[i] = (self.play_order[0]+i)%4
        self.initial_cards = [cards[0:13], cards[13:26],cards[26:39],cards[39:52]]
        self.history = []
        self.cards_sur_table=[[],[],[],[]]
        self.who_wins_each_turn = []

    def expand_history(self, card_to_add,  absolute_player):
        self.history.append(absolute_player)
        self.history.append(card_to_add)

    def judge_winner(self, cards_dans_ce_term):
        gagneur = 0
        for i in [1, 2, 3]:
            if (cards_dans_ce_term[i][0]==cards_dans_ce_term[gagneur][0]) & (ENCODING_DICT2[cards_dans_ce_term[i][1:]]>ENCODING_DICT2[cards_dans_ce_term[gagneur][1:]]):
                gagneur=i
        return gagneur

    def one_turn(self, round, robot, prophet, mBuffer, tser, device, prt):
        # label each player by 0,1,2,3
        #print("Bonjour!")
        cards_in_this_term = []
        input_vec_list=[]
        legal_choice_vec_list = []
        value_list = []
        tgt_ply_lst = []
        # transform a player's position, absolute history, and initial cards to player's state
        # then let the p-valuation network to decide which one to play
        #print("the simulation of player:", self.play_order[0])
        #print("who wins each turn", self.who_wins_each_turn)
        card_played, in_vec, target_vec, legal_choice_vec, value = tser.chercher_et_choisir(self.history,
                                                                   self.initial_cards[self.play_order[0]],
                                                                    self.play_order[0], self.cards_sur_table,
                                                                    [], self.play_order, self.who_wins_each_turn, 'A', robot, prophet, device)
        value_list.append(value)
        input_vec_list.append(in_vec)
        cards_in_this_term.append(card_played)
        legal_choice_vec_list.append(legal_choice_vec)
        tgt_ply_lst.append(target_vec)
        couleur_dans_ce_tour = card_played[0]

        #print("card played: ", card_played)
        self.cards_sur_table[self.play_order[0]].append(card_played)
        self.expand_history(card_to_vecpos(card_played), self.play_order[0])
        #print("card  played:", card_played, "history:", self.history)
        #same thing for the 2nd, 3rd, and 4th player
        for i in range(1, 4):
            #print("card in this term:", cards_in_this_term)
            #print("who wins each turn", self.who_wins_each_turn)
            card_played, in_vec, target_vec, legal_choice_vec, value = tser.chercher_et_choisir(self.history,
                                                                       self.initial_cards[self.play_order[i]],
                                                                       self.play_order[i], self.cards_sur_table,
                                                                       cards_in_this_term, self.play_order, self.who_wins_each_turn,
                                                                        couleur_dans_ce_tour, robot, prophet, device)
            tgt_ply_lst.append(target_vec)
            #print(card_played)
            self.cards_sur_table[self.play_order[i]].append(card_played)
            #print("card played:", card_played)
            self.expand_history(card_to_vecpos(card_played), self.play_order[i])
            cards_in_this_term.append(card_played)
            input_vec_list.append(in_vec)
            legal_choice_vec_list.append(legal_choice_vec)
            value_list.append(value)

        # the order of player 0 in this turn
        player_0_order=(4-self.play_order[0])%4
        # add input data into buffer for later training
        for i in range(4):
            mBuffer.add_input_sample(input_vec_list[(player_0_order+i)%4]) # input vector is pytorch tensor
            mBuffer.add_lc_sample(legal_choice_vec_list[(player_0_order+i)%4]) #legal choices vector list is np array
            mBuffer.add_target_value(value_list[(player_0_order+i)%4]) #action is an integer
            mBuffer.add_target_policy(tgt_ply_lst[(player_0_order + i) % 4])
        if prt:
            pos_dict = ['南', '东', '北', '西']
            print("第 ", round, " 轮： ", pos_dict[self.play_order[0]], " : ", cards_in_this_term[0], "; ",
                  pos_dict[self.play_order[1]], " : ", cards_in_this_term[1], "; ",
                  pos_dict[self.play_order[2]], " : ", cards_in_this_term[2], "; ",
                  pos_dict[self.play_order[3]], " : ", cards_in_this_term[3], "; ")
            #print("winner:", pos_dict[self.play_order[self.judge_winner(cards_in_this_term)]])
        # judge who wins
        winner = self.play_order[self.judge_winner(cards_in_this_term)]
        self.who_wins_each_turn.append(winner)
        self.play_order = [winner, (winner+1)%4, (winner+2)%4, (winner+3)%4]

    def calc_score(self):
        score = np.zeros(4)
        has_score_flag = [False, False, False, False]
        c10_flag = [False, False, False, False]
        heart_count=np.zeros(4)
        #calc points
        for people in range(4):
            for turn in range(13):
                if self.who_wins_each_turn[turn]==people:
                    for players in range(4):
                        pos = self.history[turn*8+players*2+1]
                        i = DECODING_DICT1[pos // 13] + DECODING_DICT2[pos % 13]
                        if i in SCORE_DICT.keys():
                            if i=="C10":
                                c10_flag[people]=True
                            else:
                                score[people]+=SCORE_DICT[i]
                                has_score_flag[people]=True
                            if i.startswith('H') or i.startswith('J'):
                                heart_count[people]+=1
            #check whole Hearts
            if heart_count[people]==13:
                score[people]+=400
            # settle transformer
            if c10_flag[people]==True:
                if has_score_flag[people]==False:
                    score[people]+=50
                else:
                    score[people]*=2
        if TRAINING:
            score[0]=score[0]+score[2]
            score[1]=score[1]+score[3]
            score[2]=score[0]
            score[3]=score[1]
        return score

    def one_round(self, robot, prophet,  mBuffer, tser, device, prt):
        self.new_shuffle()
        for no_turn in range(13):
            print("round number", no_turn)
            self.one_turn(no_turn, robot, prophet, mBuffer, tser, device, prt)

    def train(self, robot, prophet,  mBuffer, batch_size, epoch, device):
        optimizer1 = optim.Adam(robot.pnet.parameters(), lr=0.01, betas=(0.1, 0.999), eps=1e-04, weight_decay=0.0000001, amsgrad=False)
        #optimizer1 = optim.SGD(robot.parameters(), lr=0.0001, momentum=0)
        optimizer2 = optim.Adam(prophet.vnet.parameters(), lr=0.005, betas=(0.1, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)
        train_loader = torch.utils.data.DataLoader(dataset=mBuffer, batch_size=batch_size, shuffle=True)

        for jj in range(epoch):
            for i, (input_sample_i, target_policy_i, target_value_i, legal_choices_i) in enumerate(train_loader):

                target_policy_i = target_policy_i.to(device)
                legal_choices_i = legal_choices_i.to(device)
                target_value_i = target_value_i.to(device)

                loss = robot.loss_func(robot.pnet(input_sample_i), torch.tensor(target_policy_i), torch.tensor(legal_choices_i))
                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()

                loss2 = prophet.loss_func(prophet.vnet(input_sample_i), torch.tensor(target_value_i))
                optimizer2.zero_grad()
                loss2.backward()
                optimizer2.step()
            if jj%(epoch//10) == 0:
                print(jj / epoch * 100, "pourcent. P loss:", loss.item(), "V loss:", loss2.item())


    def save_model(self, robot, prophet, robot_path, prophet_path):
        torch.save(robot.pnet.state_dict(), robot_path)
        torch.save(prophet.vnet.state_dict(), prophet_path)

    def load_model(self, robot, prophet, robot_path, prophet_path):
        robot.pnet.load_state_dict(torch.load(robot_path),False)
        prophet.vnet.load_state_dict(torch.load(prophet_path),False)
        robot.pnet.eval()
        prophet.vnet.eval()
        print("model loaded")



def playMusic(filename, loops=-1, start=0.0, value=0.95):
    """    :param filename: 文件名
    :param loops: 循环次数
    :param start: 从多少秒开始播放
    :param value: 设置播放的音量，音量value的范围为0.0到1.0
    :return:
    """
    flag = False  # 是否播放过
    pygame.mixer.init()  # 音乐模块初始化

    pygame.mixer.music.load(filename)
    # pygame.mixer.music.play(loops=0, start=0.0) loops和start分别代表重复的次数和开始播放的位置。
    pygame.mixer.music.play(loops=loops, start=start)
    pygame.mixer.music.set_volume(value)  # 来设置播放的音量，音量value的范围为0.0到1.0。


print("Il commence...")
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(device)
game_batch = 10
mbuffer = Memory(52*(game_batch))
game = GameRunner()

p4 = PNet()
v4 = VNet()
#p4 = nn.DataParallel(p4)
#v4 = nn.DataParallel(v4)
#p4.cuda()
#v4.cuda()

robot_v4  = Robot(p4)
prophet_v4  = Prophet(v4)
tree_searcher = MCTS()


#game.load_model(robot_v3,  "robot-net.txt")

#game.load_model(robot_v4, prophet_v4, "robot-net.txt", "prophet-net.txt")
#print(list(robot_v2.fc2.parameters()))

#playMusic(r'D:\musique\05 Saint-Saëns VC 3 II. Andantino quasi allegretto.wav')
#pygame.mixer.music.pause()


for i in range(1000):
    for j in range(game_batch-1):
        game.one_round(robot_v4, prophet_v4, mbuffer, tree_searcher, device, False)

    game.one_round(robot_v4, prophet_v4, mbuffer, tree_searcher, device, True)
    #game.load_model(robot_v4, prophet_v4, "robot-net.txt", "prophet-net.txt")
    game.train(robot_v4, prophet_v4, mbuffer, 52*5, 40, device)
    print("李超又度过了荒废的一天，这是第", i, "天了")
    game.save_model(robot_v4, prophet_v4, "robot-net-1.txt", "prophet-net-1.txt")
    mbuffer.clear()
    #pygame.mixer.music.pause()

