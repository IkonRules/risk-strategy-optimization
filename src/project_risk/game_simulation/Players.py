import random



class Player:
    
    def __init__(self, name):
        self._name = name
        
        self._troops = 0
        self._territories = []
        self._continents = []
        self._in_game = True
        self._intelligence_dict = {}
    

    
    def battles_dict(self):
        ''' Creates battles list with attacking territory at index 0 and defending territory at index 1.
        '''
        battles_dict = {}
        index = 1
        for territory in self._territories:
            if territory._troops >= 2:
                for neighbor in territory._neighbors:
                    if neighbor._owner != self:
                        battles_dict[str(index)] = [territory, neighbor]
                        index += 1
        
        return battles_dict
    

    def choose_number_dice(self, number_dice):
        ''' Choose to roll one, two or three dice in an encounter.
        '''
        if number_dice == 1:
            roll_1 = random.randint(1, 6)
            return [roll_1]
        elif number_dice == 2:
            roll_1 = random.randint(1, 6)
            roll_2 = random.randint(1, 6)
            return [roll_1, roll_2]
        elif number_dice == 3:
            roll_1 = random.randint(1, 6)
            roll_2 = random.randint(1, 6)
            roll_3 = random.randint(1, 6)
            return [roll_1, roll_2, roll_3]
    
    
    def attacking_strategy(self):
        'Contains the attacking strategy'
        return None
    
    def defending_strategy(self):
        'Contains the defending strategy'
        return None
    
    def expansion_strategy(self):
        'Contains strategy for expansion'
        return None
    
    

''' Different player strategies for how many dice to attack with.
'''
def AttackingStrategy_low_risk(self, attacking_territory):
    return self.choose_number_dice(1)

def AttackingStrategy_high_risk(self, attacking_territory):
    if attacking_territory._troops >= 4:
        return self.choose_number_dice(3)
    elif attacking_territory._troops == 3:
        return self.choose_number_dice(2)
    elif attacking_territory._troops == 2:
        return self.choose_number_dice(1)
    

''' Different player strategies for how many dice to defend with.
'''
def DefendingStrategy_high_risk(self, defending_territory):
    if defending_territory._troops >= 2:
        return self.choose_number_dice(2)
    elif defending_territory._troops == 1:
        return self.choose_number_dice(1)

def DefendingStrategy_low_risk(self, defending_territory):
    return self.choose_number_dice(1)
             



# def ExpansionStrategy_continent_by_territories(self):
    

            

#     def attacking_strategy(self):
#         'Contains the attacking strategy'
#         return None

#     def defending_strategy(self):
#         'Contains the defending strategy'
#         return None

#     def player_strategy(self):
#         ''' To be filled with the specific strategic choices in different situations,
#         constituting a unique overall strategy.
#         '''
#         self.attacking_strategy()
#         self.defending_strategy()

# # Define specific strategies
# def aggressive_attacking_strategy(self):
#     print(f"{self._name} is using an aggressive attacking strategy!")

# def defensive_attacking_strategy(self):
#     print(f"{self._name} is focusing on defense and not attacking aggressively.")

# # Example of assigning strategies to player instances
# player1 = Player("Alice")
# player2 = Player("Bob")

# # Assign specific strategies to players
# player1.attacking_strategy = aggressive_attacking_strategy.__get__(player1, Player)
# player2.attacking_strategy = defensive_attacking_strategy.__get__(player2, Player)

# # Using the specific strategies
# player1.attacking_strategy()  # Output: Alice is using an aggressive attacking strategy!
# player2.attacking_strategy() 

# object.__get__(self, instance, owner)
    

