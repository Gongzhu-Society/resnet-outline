import numpy as np
import torch
import torch.nn.functional as F

import copy
import random
import time
import sys
sys.path.insert(0, '../')
import transforms as ts

class R18():
    def __init__(self):
        self.net = None
        self.device  =None

    def play_one_card(self, info):
        input = ts.a_standard_input(info["absolute_history"], info["my_position"], info["my_initial_cards"])
        with torch.no_grad():
            outp = self.net(torch.tensor(input.to(self.device)).unsqueeze(0))[0]
        lc = ts.trouver_les_choix_feasibles(info["my_initial_cards"], info["cards_played_in_absolute_order"][info["my_position"]], info["color_of_this_turn"])
        lcc = torch.tensor(lc).cuda()
        prob = (outp[0:52].exp()) * lcc

        _, a = torch.max(prob, 0)
        res = {"card_played": ts.pos_to_card(a),
               "value": outp[52]}
        return res
