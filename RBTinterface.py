import transforms as ts

import sys
sys.path.insert(0, 'basicrobots/')
from MrGreed import MrGreed
from MrIf import MrIf
from MrRandom import MrRandom
from mcts import MCTS
from R18 import R18
#import signal


class Robot:
    def __init__(self, name):
        self.name = name
        if name == "MrGreed":
            self.worker = MrGreed()
        elif name == "MrIf":
            self.worker = MrIf()
        elif name == "MrRandom":
            self.worker = MrRandom()
        elif name == "R18":
            self.worker = R18()
        elif name == "tree_searcher":
            self.worker = MCTS()

    def add_soul(self, net):
        self.worker.net = net
        if self.name == "tree_searcher":
            self.worker.simplified_robot.net = net

    def set_device(self, device):
        if self.name == "tree_searcher":
            self.worker.simplified_robot.device = device

    def set_imaginary_robot_list(self, imaginary_robots):
        if self.name == "tree_searcher":
            self.worker.imaginary_robot_list = imaginary_robots

    def play_one_card(self, info):
        if self.name in {"MrGreed","MrIf","MrRandom"}:
            play_order = info["play_order_in_this_turn"]
            self.worker.cards_on_table = [play_order[0]]+info["cards_played_in_this_term"]
            self.worker.cards_list = ts.whats_left(info["my_initial_cards"], info["absolute_history"], info["my_position"])
            self.worker.scores = ts.find_scores(info["absolute_history"])
            self.worker.place = info["my_position"]
            self.worker.history = ts.myhisroty_2_lchistory(info["absolute_history"])
            '''
            print("I have done calculation, here's my result")
            print(self.worker.cards_on_table)
            print(self.worker.cards_list)
            print(info["absolute_history"])
            print(self.worker.history)
            print(self.worker.place)
            print("done, I am going to play")
            '''
            card_played = self.worker.pick_a_card()
            #print("I played")
            res = {"card_played":card_played}
            return res
        elif self.name == "tree_searcher":
            return self.worker.chercher_et_choisir(info)
        elif self.name == "R18":
            return self.worker.play_one_card(info)