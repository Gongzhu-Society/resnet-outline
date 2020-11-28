#from mcts import MCTS, Node
from model import PNet, Memory
from RBTinterface import Robot
import copy

import sys
sys.path.insert(0, 'models/')
from modelsnet import P1Net

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
#from torchcontrib.optim import SWA
import random
import numpy as np
import transforms as ts

TRAINING = True

class GameRunner:
    def __init__(self):
        self.initial_cards = [[]]
        # history from judge's view
        self.history = []
        # another quick save for history, what has alresdy been played
        self.cards_sur_table = [[]]

        # play order of each turn
        self.play_order = [0, 1, 2, 3]
        # 13 elements, representing who wins each turn
        self.who_wins_each_turn = []

        # real list of 4 robots
        self.robot_list = []
        # imaginary list when performing tree search
        self.imaginary_robot_list = []
        self.imiginary_probability = []

        # which kind of player
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

    def one_turn(self, round, robot, mBuffer, device, prt):
        # label each player by 0,1,2,3
        cards_in_this_term = []
        input_vec_list=[]
        input_vec_list2 = []
        legal_choice_vec_list = []
        value_list = []
        tgt_ply_lst = []
        couleur_dans_ce_tour = "A"

        for i in range(0, 4):
            info = {"absolute_history":self.history,
                    "my_initial_cards":self.initial_cards[self.play_order[i]],
                    "my_position":self.play_order[i],
                    "cards_played_in_absolute_order":self.cards_sur_table,
                    "cards_played_in_this_term":cards_in_this_term,
                    "play_order_in_this_turn":self.play_order,
                    "who_already_won_each_turn":self.who_wins_each_turn,
                    "color_of_this_turn":couleur_dans_ce_tour,
                    "device":device}
            if robot[self.play_order[i]].name in {"tree_searcher"}:
                returned = robot[self.play_order[i]].play_one_card(info)
                tgt_ply_lst.append(returned["target_policy"])
                input_vec_list.append(returned["input_vec"])
                input_vec_list2.append(returned["v_input_vec"])
                legal_choice_vec_list.append(returned["legal_choice_vec"])
                value_list.append(returned["value"])
            else:
                returned = robot[self.play_order[i]].play_one_card(info)
            card_played = returned["card_played"]
            if i == 0:
                couleur_dans_ce_tour = card_played[0]

            self.cards_sur_table[self.play_order[i]].append(card_played)
            #print("card played:", card_played)
            self.expand_history(ts.card_to_vecpos(card_played), self.play_order[i])
            cards_in_this_term.append(card_played)

        # the order of player 0 in this turn
        #player_0_order=(4-self.play_order[0])%4
        # add input data into buffer for later training
        for i in range(len(input_vec_list)):
            mBuffer.add_input_sample(input_vec_list[i]) # input vector is pytorch tensor
            mBuffer.add_v_input_sample(input_vec_list2[i])
            mBuffer.add_lc_sample(legal_choice_vec_list[i]) #legal choices vector list is np array
            mBuffer.add_target_value(value_list[i]) #action is an integer
            mBuffer.add_target_policy(tgt_ply_lst[i])
            mBuffer.add_round(round)
        if prt:
            pos_dict = ['南', '东', '北', '西']
            print("第 ", round, " 轮： ", pos_dict[self.play_order[0]], " : ", cards_in_this_term[0], "; ",
                  pos_dict[self.play_order[1]], " : ", cards_in_this_term[1], "; ",
                  pos_dict[self.play_order[2]], " : ", cards_in_this_term[2], "; ",
                  pos_dict[self.play_order[3]], " : ", cards_in_this_term[3], "; ")

        # judge who wins
        winner = self.play_order[ts.judge_winner(cards_in_this_term)]
        self.who_wins_each_turn.append(winner)
        self.play_order = [winner, (winner+1)%4, (winner+2)%4, (winner+3)%4]

    def one_round(self, robot, mBuffer, device, prt):
        self.new_shuffle()
        for no_turn in range(13):
            if prt:
                print("round number", no_turn)
            self.one_turn(no_turn, robot,  mBuffer, device, prt)
        score = ts.calc_partial_score(self.history)
        return score


    def train(self, pnet, mBuffer, batch_size, initial_lr_1, epoch,  device1):
        optimizer1 = optim.Adam(pnet.parameters(), lr=initial_lr_1, betas=(0.9, 0.999), eps=1e-04, weight_decay=1e-4, amsgrad=False)
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

                outp = pnet(input_sample_i)
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
                    p['lr'] = initial_lr_1 *0.9


    def save_model(self, pnet,robot_path):
        torch.save(pnet.state_dict(), robot_path)
        #torch.save(prophet.vnet.state_dict(), prophet_path)

    def load_model(self, robot, prophet, robot_path, prophet_path):
        robot.pnet.load_state_dict(torch.load(robot_path),False)
        prophet.vnet.load_state_dict(torch.load(prophet_path),False)
        robot.pnet.eval()
        prophet.vnet.eval()
        print("model loaded")


#from evaluation import res_vs_if
if __name__ == '__main__':
    print("Il commence...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    game_batch = 2

    game = GameRunner()

    p4 = P1Net().to(device)
    LOAD_MODEL = True
    if LOAD_MODEL:
        #p4.load_state_dict(torch.load(r"C:\Users\SuperLi\Desktop\Shi\game\wuji\version5supervised\rawdata\robot-net-s4.txt"),False)
        p4.load_state_dict(torch.load(r"C:\Users\SuperLi\Desktop\Shi\game\wuji\version5supervised\rawdata\robot-net-s4.txt"), False)
        p4.eval()
        print('model loaded')

    treesearcher = Robot("tree_searcher")
    treesearcher.add_soul(p4)
    treesearcher.set_device(device)
    Greed = Robot("MrGreed")
    IF = Robot("MrIf")
    #robot = [treesearcher, Greed, Greed, Greed]
    robot = [treesearcher, IF, IF, IF]
    tser_ct = 0
    for single_rbt in robot:
        if single_rbt.name == "tree_searcher":
            tser_ct += 1
    mbuffer = Memory(13*tser_ct * (game_batch))

    R18_instance = Robot("R18")
    R18_instance.worker.device = device

    #imaginary_robot = [Greed, Greed, Greed, Greed]
    imaginary_robot = [IF, IF, IF, IF]
    treesearcher.set_imaginary_robot_list(imaginary_robot)

    #vis = visdom.Visdom()
    #gainline = vis.line(X=np.array([0]), Y=np.array([0]), opts=dict(showlegend=True, title='Gain over Mr.if'))
    gain_vec = []
    #game.save_model(robot_v4, prophet_v4, "robot-net.txt", "prophet-net.txt")

    for i in range(1000):
        trainingv = np.zeros((game_batch-1,4))
        for j in range(game_batch-1):
            sj = game.one_round(robot, mbuffer, device, False)
            print(sj)
            trainingv[j, :] = np.array(sj)
            if j % (game_batch//10+1) == 0:
                print("{} percent".format(j/game_batch*100))
        game.one_round(robot, mbuffer, device, True)
        trainingv = trainingv.mean(axis=0)
        doc = open("growth.txt", "a")
        print('training gain is :' + str(trainingv[0]))
        print('training gain is :' + str(trainingv[0]), file=doc)
        doc.close()

        #mbuffer=torch.load('data.pt')
        #game.load_model(robot_v4, prophet_v4, "robot-net.txt", "prophet-net.txt")

        torch.save(mbuffer, 'data.pt')
        game.train(p4, mbuffer, 16, 0.0001, 4, device)
        #result = game.trial_round()
        print("李超又度过了荒废的一天，这是第", i, "天了")
        game.save_model(p4, "robot-net.txt")
        for rbt in robot:
            if rbt.name == "tree_searcher":
                rbt.net = copy.deepcopy(p4)

        R18_instance.add_soul(p4)
        treesearcher.add_soul(p4)
        eva_rb_lst = [R18_instance, IF, R18_instance, IF]
        NOT = 100
        scr = np.zeros((NOT,4))
        for tstrn in range(NOT):
            scoret = game.one_round(eva_rb_lst, [], device, False)
            scr[tstrn,:] = np.array(scoret)
        trd = scr.mean(axis=0)

        #vis.line(X=np.array(range(len(gain_vec))), Y=np.array(gain_vec), win=gainline, opts=dict(showlegend=True, title='Gain over Mr.if'))
        doc = open("growth.txt","a")

        print('gain is :'+str(trd[0]))
        print('gain is :' + str(trd[0]), file=doc)
        doc.close()
        mbuffer.clear()

