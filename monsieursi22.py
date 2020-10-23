import numpy as np
import random
import copy

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
def vecpos_to_card(pos):
    return DECODING_DICT1[pos//13]+DECODING_DICT2[pos%13]
SCORE_DICT={'SQ':-100,'DJ':100,'C10':0,
            'H2':0,'H3':0,'H4':0,'H5':-10,'H6':-10,'H7':-10,'H8':-10,'H9':-10,'H10':-10,
            'HJ':-20,'HQ':-30,'HK':-40,'HA':-50,'JP':-60,'JG':-70}

ORDER_DICT1={'S':-300,'H':-200,'D':-100,'C':0,'J':-200}
ORDER_DICT2={'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'1':10,'J':11,'Q':12,'K':13,'A':14,'P':15,'G':16}
def cards_order(card):
    return ORDER_DICT1[card[0]]+ORDER_DICT2[card[1]]

def get_nonempty_min(l):
    if len(l)!=0:
        return len(l)
    else:
        return 100

class IfPlayer:
    def cards_left(self, initial_vec, played_vec, color_of_this_turn):
        whats_left = initial_vec - played_vec
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

        #print(pos)
        res = np.zeros(52)
        for i in range(len(pos)):
            res[pos[i]] = 1
        return res

    def initial_to_formatted(self, initialcards):
        res=np.zeros(52)
        for i in initialcards:
            res[card_to_vecpos(i)] = 1
        return res
    
    def pick_a_card(self,suit, cards_dict, cards_list, cards_on_table):
       
        try:
            assert len(cards_list)==sum([len(cards_dict[k]) for k in cards_dict])
        except:
            log("",l=3)
        #log("%s, %s, %s, %s"%(self.name,suit,self.cards_on_table,cards_list))
        #如果随便出
        if suit=="A":
            list_temp=[cards_dict[k] for k in cards_dict]
            list_temp.sort(key=get_nonempty_min)
            #log(list_temp)
            for i in range(4):
                if len(list_temp[i])==0:
                    continue
                suit_temp=list_temp[i][0][0]
                #log("thinking %s"%(suit_temp))
                if suit_temp=="S" and ("SQ" not in cards_list) \
                and ("SK" not in cards_list) and ("SA" not in cards_list):
                    choice=cards_dict["S"][-1]
                    cards_list.remove(choice)
                    cards_dict[choice[0]].remove(choice)
                    return choice
                if suit_temp=="H" and ("HQ" not in cards_list) \
                and ("HK" not in cards_list) and ("HA" not in cards_list):
                    choice=cards_dict["H"][-1]
                    cards_list.remove(choice)
                    cards_dict[choice[0]].remove(choice)
                    return choice
                if suit_temp=="C" and ("C10" not in cards_list) \
                and ("CJ" not in cards_list) and ("CQ" not in cards_list)\
                and ("CK" not in cards_list) and ("CA" not in cards_list):
                    choice=cards_dict["C"][-1]
                    cards_list.remove(choice)
                    cards_dict[choice[0]].remove(choice)
                    return choice
                if suit_temp=="D" and ("DJ" not in cards_list):
                    choice=cards_dict["D"][-1]
                    cards_list.remove(choice)
                    cards_dict[choice[0]].remove(choice)
                    return choice
            for i in range(5):
                choice=random.choice(cards_list)
                if choice not in ("SQ","SK","SA","HA","HK","C10","CJ","CQ","CK","CA","DJ"):
                    cards_list.remove(choice)
                    cards_dict[choice[0]].remove(choice)
                    return choice
        #如果是贴牌
        elif len(cards_dict[suit])==0:
            for i in ("SQ","HA","SA","SK","HK","C10","CA","HQ","HJ","CK","CQ","CJ","H10","H9","H8","H7","H6","H5"):
                if i in cards_list:
                    choice=i
                    cards_list.remove(choice)
                    cards_dict[choice[0]].remove(choice)
                    return choice
            list_temp=[cards_dict[k] for k in cards_dict]
            list_temp.sort(key=get_nonempty_min)
            for i in range(4):
                if len(list_temp[i])==0:
                    continue
                suit_temp=list_temp[i][0][0]
                choice=cards_dict[suit_temp][-1]
                cards_list.remove(choice)
                cards_dict[choice[0]].remove(choice)
                return choice
        #如果只有这一张
        elif len(cards_dict[suit])==1:
            choice=cards_dict[suit][-1]
            cards_list.remove(choice)
            cards_dict[choice[0]].remove(choice)
            return choice

        #如果是猪并且剩好几张猪牌
        if suit=="S":
            if ("SQ" in cards_list) and (("SK" in cards_on_table) or ("SA" in cards_on_table)):
                choice="SQ"
                cards_list.remove(choice)
                cards_dict[choice[0]].remove(choice)
                return choice
            if len(cards_on_table)==4 and ("SQ" not in cards_on_table):
                choice=cards_dict["S"][-1]
                if choice=="SQ":
                    choice=cards_dict["S"][-2]
                cards_list.remove(choice)
                cards_dict[choice[0]].remove(choice)
                return choice
            else:
                if "SA" in cards_on_table[1:]:
                    max_pig=cards_order("SA")
                elif "SK" in cards_on_table[1:]:
                    max_pig=cards_order("SK")
                else:
                    max_pig=cards_order("SQ")
                for i in cards_dict["S"][::-1]:
                    if cards_order(i)<max_pig:
                        choice=i
                        cards_list.remove(choice)
                        cards_dict[choice[0]].remove(choice)
                        return choice
                else:
                    choice=cards_dict["S"][-1]
                    cards_list.remove(choice)
                    cards_dict[choice[0]].remove(choice)
                    return choice
        #如果是变压器并且草花剩两张以上
        if suit=="C":
            if ("C10" in cards_list)\
            and (("CJ" in cards_on_table) or ("CQ" in cards_on_table) or\
                 ("CK" in cards_on_table) or ("CA" in cards_on_table)):
                choice="C10"
                cards_list.remove(choice)
                cards_dict[choice[0]].remove(choice)
                return choice
            if len(cards_on_table)==4 and ("C10" not in cards_on_table):
                choice=cards_dict["C"][-1]
                if choice=="C10":
                    choice=cards_dict["C"][-2]
                cards_list.remove(choice)
                cards_dict[choice[0]].remove(choice)
                return choice
            else:
                if "CA" in cards_on_table[1:]:
                    max_club=cards_order("CA")
                elif "CK" in cards_on_table[1:]:
                    max_club=cards_order("CK")
                elif "CQ" in cards_on_table[1:]:
                    max_club=cards_order("CQ")
                elif "CJ" in cards_on_table[1:]:
                    max_club=cards_order("CJ")
                else:
                    max_club=cards_order("C10")
                for i in cards_dict["C"][::-1]:
                    if cards_order(i)<max_club:
                        choice=i
                        cards_list.remove(choice)
                        cards_dict[choice[0]].remove(choice)
                        return choice
                else:
                    choice=cards_dict["C"][-1]
                    cards_list.remove(choice)
                    cards_dict[choice[0]].remove(choice)
                    return choice
        #如果是羊并且剩两张以上
        if suit=="D":
            if len(cards_on_table)==4 and ("DJ" in cards_dict["D"])\
            and ("DA" not in cards_on_table) and ("DK" not in cards_on_table)\
            and ("DQ" not in cards_on_table):
                choice="DJ"
                cards_list.remove(choice)
                cards_dict[choice[0]].remove(choice)
                return choice
            choice=cards_dict["D"][-1]
            if choice=="DJ":
                choice=cards_dict["D"][-2]
            cards_list.remove(choice)
            cards_dict[choice[0]].remove(choice)
            return choice
        #如果是红桃
        if suit=="H":
            max_heart=-1000
            for i in cards_on_table[1:]:
                if i[0]=="H" and cards_order(i)>max_heart:
                    max_heart=cards_order(i)
            for i in cards_dict["H"][::-1]:
                if cards_order(i)<max_heart:
                    choice=i
                    cards_list.remove(choice)
                    cards_dict[choice[0]].remove(choice)
                    return choice
        #log("cannot be decided by rules")
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
        if res!='RIEN':
            return res

        prb = legal_choices/np.sum(legal_choices)
        sample_output = np.random.multinomial(1, prb, size=1)
        the_card = vec_to_card(sample_output[0])
        return the_card