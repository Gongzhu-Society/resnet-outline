from mcts import MCTS, Node
from model import PNet, VNet, Robot, Prophet, Memory
import sys
from modelsnet import P1Net

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
#from torchcontrib.optim import SWA
import random
import numpy as np



ENCODING_DICT1={'H':0, 'C':1, 'D':2, 'S':3}
ENCODING_DICT2={'2':0, '3':1, '4':2, '5':3, '6':4, '7':5, '8':6, '9':7, '10':8, 'J':9, 'Q':10, 'K':11, 'A':12}
DECODING_DICT1=['H', 'C', 'D', 'S']
DECODING_DICT2=['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
def card_to_vecpos(card):
    return ENCODING_DICT1[card[0]] * 13 + ENCODING_DICT2[card[1:]]

def vec_to_card(vec):
    pos = np.where(vec == 1)[0][0]
    return DECODING_DICT1[pos//13]+DECODING_DICT2[pos%13]


SCORE_DICT={'SQ':-100,'DJ':100,'C10':0,
            'H2':0,'H3':0,'H4':0,'H5':-10,'H6':-10,'H7':-10,'H8':-10,'H9':-10,'H10':-10,
            'HJ':-20,'HQ':-30,'HK':-40,'HA':-50,'JP':-60,'JG':-70}
TRAINING = True

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
        cards_in_this_term = []
        input_vec_list=[]
        input_vec_list2 = []
        legal_choice_vec_list = []
        value_list = []
        tgt_ply_lst = []
        # transform a player's position, absolute history, and initial cards to player's state
        # then let the p-valuation network to decide which one to play
        #print("the simulation of player:", self.play_order[0])
        #print("who wins each turn", self.who_wins_each_turn)
        card_played, in_vec, target_vec, legal_choice_vec, v_in_vec, value = tser.chercher_et_choisir(self.history,
                                                                   self.initial_cards[self.play_order[0]],
                                                                    self.play_order[0], self.cards_sur_table,
                                                                    [], self.play_order, self.who_wins_each_turn, 'A', robot, prophet, device)
        value_list.append(value)
        input_vec_list.append(in_vec)
        input_vec_list2.append(v_in_vec)
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
            card_played, in_vec, target_vec, legal_choice_vec, v_in_vec, value = tser.chercher_et_choisir(self.history,
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
            input_vec_list2.append(v_in_vec)
            legal_choice_vec_list.append(legal_choice_vec)
            value_list.append(value)

        # the order of player 0 in this turn
        player_0_order=(4-self.play_order[0])%4
        # add input data into buffer for later training
        for i in range(4):
            mBuffer.add_input_sample(input_vec_list[(player_0_order+i)%4]) # input vector is pytorch tensor
            mBuffer.add_v_input_sample(input_vec_list2[(player_0_order + i) % 4])
            mBuffer.add_lc_sample(legal_choice_vec_list[(player_0_order+i)%4]) #legal choices vector list is np array
            mBuffer.add_target_value(value_list[(player_0_order+i)%4]) #action is an integer
            mBuffer.add_target_policy(tgt_ply_lst[(player_0_order + i) % 4])
            mBuffer.add_round(round)
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

    def train(self, robot, prophet,  mBuffer, batch_size, initial_lr_1, epoch,  device1):
        optimizer1 = optim.Adam(robot.pnet.parameters(), lr=initial_lr_1, betas=(0.9, 0.999), eps=1e-04, weight_decay=1e-4, amsgrad=False)
        #optimizer1 = base_optimizer1 #torchcontrib.optim.SWA(base_optimizer1, swa_start=1, swa_freq=5, swa_lr=initial_lr_1)
        #optimizer1 = optim.SGD(robot.pnet.parameters(), lr=initial_lr_1, momentum=0.9)
        #optimizer2 = optim.Adam(prophet.vnet.parameters(), lr=initial_lr_2, betas=(0.1, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)
        train_loader = torch.utils.data.DataLoader(dataset=mBuffer, batch_size=batch_size, shuffle=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for jj in range(epoch):
            for i, (input_sample_i, v_input_i, target_policy_i, target_value_i, legal_choices_i, round_i) in enumerate(train_loader):

                target_policy_i = target_policy_i.to(device)
                legal_choices_i = legal_choices_i.to(device)
                target_value_i = target_value_i.to(device)
                input_sample_i = input_sample_i.to(device)

                outp = robot.pnet(input_sample_i)
                features = outp[:,:52]* torch.tensor(legal_choices_i).clone().detach()
                probs = F.softmax(features, dim=1)
                loss1 = F.kl_div(probs.log(), torch.tensor(target_policy_i).clone().detach(), reduction="batchmean")

                loss2 = (outp[:,52]-target_value_i).norm(2)
                loss = loss1+loss2*0.01

                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()


            if jj%(epoch//10+1) == 0:
                print(jj / epoch * 100, "pourcent. P loss:", loss1.item(), "V loss:", loss2.item())
                for p in optimizer1.param_groups:
                    p['lr'] = initial_lr_1 / np.sqrt(1 + jj/10)

        #optimizer1.swap_swa_sgd()

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

from arena import res_vs_if
if __name__ == '__main__':
    print("Il commence...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    game_batch = 25
    mbuffer = Memory(52*(game_batch))
    game = GameRunner()

    p4 = P1Net().to(device)
    LOAD_MODEL = 1
    if LOAD_MODEL == 1:
        #p4.load_state_dict(torch.load(r"C:\Users\SuperLi\Desktop\Shi\game\wuji\version5supervised\rawdata\robot-net-s4.txt"),False)
        p4.load_state_dict(torch.load(r"robot-net-s4.txt"), False)
        p4.eval()
        print('model loaded')
    elif LOAD_MODEL == 2:
        p4.load_state_dict(torch.load(r"robot-net-s0.txt"), False)
        p4.eval()
        print('model loaded')
    v4 = VNet()

    robot_v4  = Robot(p4)
    prophet_v4  = Prophet(v4)
    tree_searcher = MCTS(p4)
    #vis = visdom.Visdom()
    #gainline = vis.line(X=np.array([0]), Y=np.array([0]), opts=dict(showlegend=True, title='Gain over Mr.if'))
    gain_vec = []
    #game.save_model(robot_v4, prophet_v4, "robot-net.txt", "prophet-net.txt")

    for i in range(1000):

        for j in range(game_batch-1):
            game.one_round(robot_v4, prophet_v4, mbuffer, tree_searcher, device, False)
        game.one_round(robot_v4, prophet_v4, mbuffer, tree_searcher, device, True)

        #mbuffer=torch.load('data.pt')
        #game.load_model(robot_v4, prophet_v4, "robot-net.txt", "prophet-net.txt")

        torch.save(mbuffer, 'data.pt')
        game.train(robot_v4, prophet_v4, mbuffer, 16, 0.0001, 4, device)
        #result = game.trial_round()
        print("李超又度过了荒废的一天，这是第", i, "天了")
        game.save_model(robot_v4, prophet_v4, "robot-net-s0.txt", "prophet-net.txt")

        score0, score1 = res_vs_if(p4, 200)
        gain = (score1-score0).mean()
        gain_vec.append(gain)
        #vis.line(X=np.array(range(len(gain_vec))), Y=np.array(gain_vec), win=gainline, opts=dict(showlegend=True, title='Gain over Mr.if'))
        doc = open("growth.txt","a")
        print('gain is :'+str(gain))
        print('gain is :' + str(gain),file=doc)
        mbuffer.clear()

