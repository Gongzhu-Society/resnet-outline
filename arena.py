import numpy as np
import random
import sys
import torch

ENCODING_DICT1={'H':0, 'C':1, 'D':2, 'S':3}
ENCODING_DICT2={'2':0, '3':1, '4':2, '5':3, '6':4, '7':5, '8':6, '9':7, '10':8, 'J':9, 'Q':10, 'K':11, 'A':12}
DECODING_DICT1=['H', 'C', 'D', 'S']
DECODING_DICT2=['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
def card_to_vecpos(card):
    #print(card)
    return ENCODING_DICT1[card[0]] * 13 + ENCODING_DICT2[card[1:]]
def vec_to_card(vec):
    #pos=vec.index(1)
    pos = np.where(vec == 1)[0][0]
    return DECODING_DICT1[pos//13]+DECODING_DICT2[pos%13]

SCORE_DICT={'SQ':-100,'DJ':100,'C10':0,
            'H2':0,'H3':0,'H4':0,'H5':-10,'H6':-10,'H7':-10,'H8':-10,'H9':-10,'H10':-10,
            'HJ':-20,'HQ':-30,'HK':-40,'HA':-50,'JP':-60,'JG':-70}

class GameRunner:
    def __init__(self):
        self.initial_cards = [[]]
        # history from judge's view
        self.history = np.zeros((4, 13, 52+4))
        self.new_form_history = []
        # another quick save for history
        self.cards_sur_table = [[]]

        self.play_order = [0, 1, 2, 3]
        # 13 elements, representing who wins each turn
        self.who_wins_each_turn = []

        self.robot = []


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
        self.history = np.zeros((4, 13, 52 + 4))
        self.cards_sur_table=[[],[],[],[]]
        self.who_wins_each_turn = []

    def expand_history(self, card_to_add, pos_in_this_turn, absolute_player, which_round):
        self.history[absolute_player, which_round, 4 + card_to_vecpos(card_to_add)] = 1
        self.history[absolute_player, which_round, pos_in_this_turn] = 1

    def initial_to_formatted(self, initialcards):
        res=np.zeros(52)
        for i in initialcards:
            res[card_to_vecpos(i)] = 1
        return res

    def to_state(self, play_order, absolute_history, initial_cards):
        res = absolute_history[play_order].flatten()
        res = np.concatenate((res, absolute_history[(play_order+1)%4].flatten()))
        res = np.concatenate((res, absolute_history[(play_order + 2) % 4].flatten()))
        res = np.concatenate((res, absolute_history[(play_order + 3) % 4].flatten()))
        res = np.concatenate((res, self.initial_to_formatted(initial_cards)))
        return res

    def judge_winner(self, cards_dans_ce_term):
        gagneur = 0
        for i in [1, 2, 3]:
            if (cards_dans_ce_term[i][0]==cards_dans_ce_term[gagneur][0]) & (ENCODING_DICT2[cards_dans_ce_term[i][1:]]>ENCODING_DICT2[cards_dans_ce_term[gagneur][1:]]):
                gagneur=i
        return gagneur

    def one_turn(self, round, prt):
        # label each player by 0,1,2,3
        cards_in_this_term = []
        cards_in_this_term_li = []
        input_vec_list=[]
        legal_choice_vec_list = []
        # transform a player's position, absolute history, and initial cards to player's state
        # then let the p-valuation network to decide which one to play

        #state_vec1 = self.to_state(self.play_order[0], self.history, self.initial_cards[self.play_order[0]])
        card_played = self.robot[self.play_order[0]].play_one_card([round, cards_in_this_term_li, self.play_order,self.new_form_history,self.cards_sur_table],
                                                                   self.play_order[0], self.history, self.initial_cards[self.play_order[0]], self.cards_sur_table[self.play_order[0]], 'A', False)
        #print(self.play_order[0], card_played)
        #input_vec_list.append(in_vec)
        cards_in_this_term_li.append(self.play_order[0])
        cards_in_this_term_li.append(card_played)
        cards_in_this_term.append(card_played)
        #legal_choice_vec_list.append(legal_choice_vec)
        couleur_dans_ce_tour = card_played[0]
        #print("card played: ", card_played)
        self.cards_sur_table[self.play_order[0]].append(card_played)
        self.expand_history(card_played, 0, self.play_order[0], round)
        self.new_form_history.append(self.play_order[0])
        self.new_form_history.append(card_to_vecpos(card_played))
        #same thing for the 2nd, 3rd, and 4th player
        for i in range(1, 4):
            #print("player: ", self.play_order[i])
            #state_vec1 = self.to_state(self.play_order[i], self.history, self.initial_cards[self.play_order[i]])
            card_played = self.robot[self.play_order[i]].play_one_card([round, cards_in_this_term_li, self.play_order,self.new_form_history,self.cards_sur_table],
                                                                       self.play_order[i], self.history, self.initial_cards[self.play_order[i]],
                                              self.cards_sur_table[self.play_order[i]], couleur_dans_ce_tour,False)
            #print(self.play_order[i], card_played)
            self.cards_sur_table[self.play_order[i]].append(card_played)
            #print("card played:", card_played)
            self.expand_history(card_played, i, self.play_order[i], round)
            cards_in_this_term.append(card_played)
            cards_in_this_term_li.append(card_played)
            self.new_form_history.append(self.play_order[i])
            self.new_form_history.append(card_to_vecpos(card_played))
            #input_vec_list.append(in_vec)
            #legal_choice_vec_list.append(legal_choice_vec)

        # the order of player 0 in this turn
        player_0_order=(4-self.play_order[0])%4
        # add input data into buffer for later training

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
                        i=vec_to_card(self.history[players, turn, 4:])
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

        score[0]=score[0]+score[2]
        score[1]=score[1]+score[3]
        score[2]=score[0]
        score[3]=score[1]
        return score

    def one_round(self, prt):
        self.new_shuffle()
        for no_turn in range(13):
            self.one_turn(no_turn, prt)
        #print("Hello here")
        result = self.calc_score()
        self.cards_sur_table=[[],[],[],[]]
        self.history=np.zeros((4, 13, 52+4))
        self.new_form_history=[]
        return result

#from .interfacesun2 import missmcts

from monsieursi22 import IfPlayer
#sys.path.insert(0, '../../../')
#from version5supervised.rawdata.modelsnet import P1Net

def res_vs_if(resnet,NOR):
    game = GameRunner()

    robot0 = IfPlayer()
    robot2 = IfPlayer()
    # robot0 = RandomPlayer()
    # robot2 = RandomPlayer()

    # robot1 = RandomPlayer()
    # robot3 = RandomPlayer()
    # robot1 = IfPlayer()
    # robot3 = IfPlayer()
    #robot1 = missmcts()
    #robot3 = missmcts()
    robot1 = misss(resnet)
    robot3 = misss(resnet)

    game.robot = [robot0, robot1, robot2, robot3]
    res02 = np.zeros(NOR)
    res13 = np.zeros(NOR)

    for i in range(NOR):
        result = game.one_round(False)
        res13[i] = result[1]
        res02[i] = result[0]
    return res02, res13

if __name__ == '__main__':
    game = GameRunner()

    robot0 = IfPlayer()
    robot2 = IfPlayer()
    #robot0 = RandomPlayer()
    #robot2 = RandomPlayer()

    #robot1 = RandomPlayer()
    #robot3 = RandomPlayer()
    #robot1 = IfPlayer()
    #robot3 = IfPlayer()
    resnet = P1Net().cuda()
    resnet.load_state_dict(
        torch.load(r"robot-net-s4.txt"), False)
    resnet.eval()
    #robot1 = missmcts(resnet)
    #robot3 = missmcts(resnet)
    robot1 = misss(resnet)
    robot3 = misss(resnet)

    game.robot = [robot0, robot1, robot2, robot3]
    doc = open('result.txt','w')
    print("南北","东西",file=doc)
    NOR = 200
    res = np.zeros(NOR)
    res02 = np.zeros(NOR)
    res13 = np.zeros(NOR)

    for i in range(NOR):
        if i%(NOR//10)==0:
            result = game.one_round(True)
        else:
            result = game.one_round(False)
        res[i] = result[1]-result[0]
        res13[i] = result[1]
        res02[i] = result[0]
        print(result[0], result[1], file=doc)
        if i%10==3 :
            print("刚刚，颓废的机器人打了", i, "盘")
    doc.close()
    doc = open("stats.txt",'w')
    print("number of rounds",NOR,file=doc)
    print("average gain",np.mean(res),file=doc)
    print("gain std",np.std(res)/np.sqrt(NOR-1),file=doc)
    print("02 average",np.mean(res02),file=doc)
    print("13 average",np.mean(res13),file=doc)
    doc.close()
