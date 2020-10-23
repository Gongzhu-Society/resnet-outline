import numpy as np
import torch
import torch.nn.functional as F
#from mcts import MCTS, Node
#from modelsnet import PNet, P2Net, P1Net, Robot, Memory
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


class misss():
    def __init__(self, pnet):

        self.p4 = pnet


    #history_real, ini_card, my_position, cards_already_played, this_turn_till_me,
    #first_play_order, who_wins_each_turn, couleur, robot, prophet, device)
    def play_one_card(self, info, my_position, history_real, ini_card, cards_already_played, couleur, training):
        #print('player', my_position, 'inicards are', ini_card)
        input = a_standard_input(history_real, my_position, ini_card)
        with torch.no_grad():
            outp = self.p4(torch.tensor(input).unsqueeze(0).cuda())[0]
            lc = trouver_les_choix_feasibles(ini_card, cards_already_played, couleur)
            lcc = torch.tensor(lc).cuda()
            prob = (outp[0:52].exp())*lcc
            #prob = F.softmax(prob, dim=0)
            #prob = prob*lcc
            _, a = torch.max(prob,0)
            #print('player',my_position, 'plays',a)
            return pos_to_card(a), outp[52]
#[[], this_turn_till_me], self.play_order[j], history, hidden_information[self.play_order[j]],
#                                              self.cards_played_original[self.play_order[j]], color_of_this_turn, True)