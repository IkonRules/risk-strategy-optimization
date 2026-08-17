import random
import sys
from collections import deque
from project_risk.game_simulation.Players import Player
from project_risk.game_simulation import Board
import os
from project_risk.game_simulation.Players import AttackingStrategy_high_risk
from project_risk.game_simulation.Players import AttackingStrategy_low_risk
from project_risk.game_simulation.Players import DefendingStrategy_high_risk
from project_risk.game_simulation.Players import DefendingStrategy_low_risk

player1 = Player('Player1')
player2 = Player('Player2')
player3 = Player('Player3')

player1.attacking_strategy = AttackingStrategy_high_risk.__get__(player1, Player)
player1.defending_strategy = DefendingStrategy_high_risk.__get__(player1, Player)

player2.attacking_strategy = AttackingStrategy_low_risk.__get__(player2, Player)
player2.defending_strategy = DefendingStrategy_low_risk.__get__(player2, Player)

player3.attacking_strategy = AttackingStrategy_high_risk.__get__(player3, Player)
player3.defending_strategy = DefendingStrategy_high_risk.__get__(player3, Player)


players_list = [player1, player2]
active_players_list = players_list



def close_program():
    ''' Supportive function to shut down program in a controlled way,
    taking a snapshot of the gameboard before termination.
    '''
    print("Program is terminating...")
    save_game_state_to_image(active_players_list, output_image_path=3)
    sys.exit(0)




def print_battles_dict(battles_dict):
    ''' Prints the battles dict items.
    '''
    print('Battles dict:')
    for key, value in battles_dict.items():
        print(value[0]._name, value[1]._name)




def initialize_playing_queue(players_list, verbose=False):
    ''' Chooses starting player and creates queue for the game 
    based on the starting player.
    '''
    
    # Choose starting player at random.
    starting_player = players_list[random.randint(0, len(players_list) - 1)]
    
    if verbose:
        print(starting_player._name)
    
    # Initialize playing queue, add starting player then players to the right, then from left to right.
    playing_queue = deque()
    playing_queue.append(starting_player)

    for player in players_list:
        if players_list.index(player) > players_list.index(starting_player):
            playing_queue.append(player)
    for player in players_list:
        if players_list.index(player) < players_list.index(starting_player):
            playing_queue.append(player)
    
    return playing_queue




def place_troops(playing_queue, verbose=False):
    ''' Assigns territories to the players and places troops on to the board,
    based on a given order.
    '''
    
    # Initial parameter values.
    available_territories = Board.all_territories_list.copy()
    all_available_troops = 0
    available_troops = {player._name: 40 for player in players_list}
    for value in available_troops:
        all_available_troops += available_troops[value]
    
    # Assign free territory and place troops for player first in queue.
    while available_territories != []:
        player = playing_queue.popleft()
        territory = available_territories[random.randint(0, len(available_territories) - 1)]
        player._territories.append(territory)
        available_troops[str(player._name)] += (-1)
        all_available_troops += (-1)
        territory._owner = player
        territory._troops += 1
        playing_queue.append(player)
        available_territories.remove(territory)
        
    # Print messages for debugging.
        if verbose:
            print(f'{player._name} has aquired {territory._name}!')
            print(f'{player._name} has {available_troops[str(player._name)]} available troops left!')
    if verbose:
        print(f'Number of available territories left is {len(available_territories)}!')
        print(available_troops)
    
    # Place remaning troops in territories controlled by players.
    while all_available_troops != 0:
        player = playing_queue.popleft()
        territory = player._territories[random.randint(0, len(player._territories) - 1)]
        available_troops[str(player._name)] += (-1)
        all_available_troops += (-1)
        territory._troops += 1
        playing_queue.append(player)
        
        if verbose:
            print(f'{player._name} has placed a troop on {territory._name}!')
            print(f'{player._name} has {available_troops[str(player._name)]} available troops left!')
    
    


def count_continent_points(player):
    ''' Counts number of extra troops given each round to a player for owning continents.
    '''
    new_troops_based_on_continents = 0 
    if 'Nort America' in player._continents:
        new_troops_based_on_continents += 5
    if 'South America' in player._continents:
        new_troops_based_on_continents += 2
    if 'Africa' in player._continents:
        new_troops_based_on_continents += 3
    if 'Europe' in player._continents:
        new_troops_based_on_continents += 5
    if 'Asia' in player._continents:
        new_troops_based_on_continents += 7
    if 'Australia' in player._continents:
        new_troops_based_on_continents += 2
    return new_troops_based_on_continents




def add_troop_reinforcements(attacking_player):
    ''' Adds troops to a random territory in battles list one at a time.
    '''
    battles_dict = attacking_player.battles_dict()
    move_troops(attacking_player)
    
    # Calculate number of new troops.
    new_troops_based_on_territories = int(len(attacking_player._territories) / 3)
    new_troops_based_on_continents = count_continent_points(attacking_player)
    number_new_troops = new_troops_based_on_territories + new_troops_based_on_continents
    
    if battles_dict != {}:
        len_battles_dict = len(battles_dict)
        for i in range(1, number_new_troops + 1):
            random_battle_territory_index = random.randint(1, len_battles_dict)
            territory_to_add_troops = battles_dict[str(random_battle_territory_index)][0]
            territory_to_add_troops._troops += 1
    else:
        territories_list = attacking_player._territories
        len_territories_list = len(territories_list)
        for i in range(1, number_new_troops + 1):
            random_battle_territory_index = random.randint(0, len_territories_list - 1)
            territory_to_add_troops = territories_list[random_battle_territory_index]
            territory_to_add_troops._troops += 1




def move_troops_after_battle(attacking_territory, defending_territory):
    ''' Moves the all except one troops from an attacking territory,
    to the attacked territory if battle successful.
    '''
    attacking_territory_troops_number = attacking_territory._troops
    if defending_territory._troops < 1:
        defending_territory._troops = attacking_territory_troops_number - 1
        attacking_territory._troops = 1
        print(f'{defending_territory._troops} troops has been moved to {defending_territory._name}')




def move_troops(attacking_player):
    ''' Moves troops from "non-battle" territories to battle territories
    in order for the game to continue.
    '''
    battles_dict = attacking_player.battles_dict()
    print_battles_dict(battles_dict)
    print(f'Length of {attacking_player._name}s battles dict is :', len(battles_dict))
    
    # Gather all available troops from non-battle territories.
    total_nr_troops_to_move = 0
    for territory in attacking_player._territories:
        if territory._troops >= 2 and territory not in battles_dict:
            total_nr_troops_to_move += territory._troops - 1  
            territory._troops = 1
    print(f'Total nr of troops to move for {attacking_player._name} is :', total_nr_troops_to_move)
    
    if battles_dict != {}:
        for i in range(1, total_nr_troops_to_move + 1):
            len_battles_dict = len(battles_dict)
            random_battle_territory_index = random.randint(1, len_battles_dict)
            # print('before: ', battles_dict[str(random_battle_territory_index)][0]._name, battles_dict[str(random_battle_territory_index)][0]._troops)
            battles_dict[str(random_battle_territory_index)][0]._troops += 1
            # print('after :', battles_dict[str(random_battle_territory_index)][0]._name, battles_dict[str(random_battle_territory_index)][0]._troops)
    else:
        for i in range(1, total_nr_troops_to_move + 1):
            len_territories_list = len(attacking_player._territories)
            random_territory_index = random.randint(0, len_territories_list - 1)
            attacking_player._territories[random_territory_index]._troops += 1




def change_territory_ownership(attacking_player, defending_player, territory):
    ''' Change ownership of a territory if attack on territory is successful..
    '''
    defending_player._territories.remove(territory)
    territory._owner = None
    territory._owner = attacking_player
    attacking_player._territories.append(territory)
    print(f'{attacking_player._name} now owns {territory._name}!')




def update_player_continents(player):
    ''' Updates the continents owned by player.
    '''
    for continent, territories in Board.continent_dict.items():
        if all(territory in player._territories for territory in territories):
            if continent not in player._continents:
                player._continents.append(continent)
                print(f'{player._name} now owns all of {continent}!')
        else:
            if continent in player._continents:
                player._continents.remove(continent)
                print(f'{player._name} has lost controll over {continent}!')




def update_all_players_continents(attacking_player, defending_player):
    ''' Updates continents for players after a battle.
    '''
    update_player_continents(attacking_player)
    update_player_continents(defending_player)




def determine_if_player_in_game(player):
    ''' Determine if player is still in game by counting total number of troops.
    '''
    if len(player._territories) == 0:
        player._in_game = False
    return player._in_game




def determine_battle_continuation(attacking_territory, defending_territory):
    ''' Determine if to continue battle based on number of troops in 
    attacking territory and territory being attacked.
    '''
    continue_battle = True
    nr_attacking_troops = attacking_territory._troops
    nr_defending_troops = defending_territory._troops
    
    if nr_attacking_troops < 2 or nr_defending_troops < 1:
        continue_battle = False
    else:
        print('Battle continues!')

    return continue_battle




def determine_battle_winner(attacking_territory, defending_territory, attacking_player, defending_player):
    ''' Determine and return winner after battle.
    '''
    winner = None
    if attacking_territory._troops < 2:
        winner = defending_player
        print(f'{attacking_player._name} yields the battle!')
    if defending_territory._troops < 1:
        winner = attacking_player
        print(f'{defending_player._name} is out of troops!')
    
    return winner




def compare_two_dice(attacking_dice, defending_dice):
    ''' Compares two individual dice to return winner. Used in other functions. 
    '''
    winner = None 
    if attacking_dice > defending_dice:
         winner = 'Attacker'
    else:
         winner = 'Defender'
    return winner
    
   
    


def determine_dice_outcome(attacking_player, defending_player, attacking_territory, defending_territory):
    ''' Rolls attacking and defending dice based on player strategies and 
    counts number of troops slain of each player to return as a list.
    '''
    # Initialize and sort necessary variables and lists.
    attacking_dice_list = attacking_player.attacking_strategy(attacking_territory)
    defending_dice_list = defending_player.defending_strategy(defending_territory)
    
    attacking_sorted_reversed_list = sorted(attacking_dice_list)[::-1]
    defending_sorted_reversed_list = sorted(defending_dice_list)[::-1]
    
    attacking_dice_number = len(attacking_dice_list)
    defending_dice_number = len(defending_dice_list)
    
    attacking_troops_killed = 0
    defending_troops_killed = 0
    
    # Compare the correct dice in each players rolls to determine troops killed.
    if (attacking_dice_number == 3 and defending_dice_number == 2) or (attacking_dice_number == 2 and defending_dice_number == 2):
        winner_first_encounter = compare_two_dice(attacking_sorted_reversed_list[0], defending_sorted_reversed_list[0])
        winner_second_encounter = compare_two_dice(attacking_sorted_reversed_list[1], defending_sorted_reversed_list[1])
        if winner_first_encounter == 'Attacker':
            defending_troops_killed += 1
        else:
            attacking_troops_killed += 1
        if winner_second_encounter == 'Attacker':
            defending_troops_killed += 1
        else:
            attacking_troops_killed += 1
    else:
        winner_first_encounter = compare_two_dice(attacking_sorted_reversed_list[0], defending_sorted_reversed_list[0])
        if winner_first_encounter == 'Attacker':
            defending_troops_killed += 1
        else:
            attacking_troops_killed += 1
    
    return [attacking_troops_killed, defending_troops_killed]
    




def battle(attacking_player):
    ''' Simulates battle and return loser to determine if out of game.
    '''
    # Initialize necessary parameters to simulate a battle.
    battles_dict = attacking_player.battles_dict()
    len_battles_dict = len(battles_dict)
    random_battle_index = random.randint(1, len_battles_dict)
    attacking_territory = battles_dict[str(random_battle_index)][0]
    defending_territory = battles_dict[str(random_battle_index)][1]
    defending_player = defending_territory._owner
    
    # Update number of troops after each battle encounter.
    while determine_battle_continuation(attacking_territory, defending_territory):
        print(f'{attacking_player._name} attacks {defending_territory._name} from {attacking_territory._name}!')
        encounter_result = determine_dice_outcome(attacking_player, defending_player, 
                                                  attacking_territory, defending_territory)
        attacking_territory._troops += (- encounter_result[0])
        defending_territory._troops += (- encounter_result[1])
        print(f'{encounter_result[0]} attacking troops have been slain!')
        print(f'{encounter_result[1]} defending troops have been slain!')
        print(f'{attacking_territory._name} has {attacking_territory._troops} left!')
        print(f'{defending_territory._name} has {defending_territory._troops} left!')
    
    # Determine winner and loser and return loser.
    winner = determine_battle_winner(attacking_territory, defending_territory, attacking_player, defending_player)
    loser = None
    if winner == attacking_player:
        change_territory_ownership(attacking_player, defending_player, defending_territory)
        move_troops_after_battle(attacking_territory, defending_territory)
        loser = defending_player
    else:
        loser = attacking_player
    
    update_all_players_continents(attacking_player, defending_player)
    print(f'{winner._name} winns the battle!\n')
        
    return loser



    
def save_game_state_to_image(players_list, 
                             file_path=None,
                             output_image_path=None,
                             show_coords=False):
    """Render a board state using caller-supplied image paths.

    Pillow is optional and imported only when rendering is requested. The
    research copy used machine-local default paths; public callers must provide
    both paths explicitly.
    """
    if file_path is None or output_image_path is None:
        raise ValueError("file_path and output_image_path must be provided")
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise ImportError("Pillow is required for board rendering") from exc
    
    # Load the game board image
    image = Image.open(file_path)
    draw = ImageDraw.Draw(image)

    # Define colors for each player
    player_colors = {
        "Player1": (255, 0, 0),  # Red
        "Player2": (0, 255, 0),  # Green
        "Player3": (0, 0, 255),  # Blue
        "Player4": (255, 255, 0),  # Yellow
        # Add more players and colors if necessary
    }

    # # Debug: Print player names and check if they match
    # for player in players_list:
    #     print(f"Player: {player._name}")

    # Define a font
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except IOError:
        font = ImageFont.load_default()

    # Updated coordinates for placing text on the territories
    territory_coords = {
        'Alaska': (70, 90),
        'Northwest Territory': (210, 90),
        'Greenland': (380, 30),
        'Alberta': (160, 180),
        'Ontario': (220, 180),
        'Quebeck': (310, 180),
        'Western United States': (180, 250),
        'Eastern United States': (280, 260),
        'Central America': (180, 350),
        'Venezuela': (250, 440),
        'Brazil': (360, 500),
        'Peru': (280, 560),
        'Argentina': (270, 700),
        'North Africa': (540, 500),
        'Egypt': (620, 460),
        'East Africa': (680, 540),
        'Congo': (620, 600),
        'South Africa': (630, 700),
        'Madagascar': (730, 700),
        'Iceland': (490, 130),
        'Scandinavia': (600, 100),
        'Great Britain': (460, 220),
        'Northern Europe': (580, 240),
        'Ukraine': (700, 180),
        'Western Europe': (480, 350),
        'Southern Europe': (580, 320),
        'Siberia': (850, 40),
        'Yakutsk': (960, 60),
        'Kamchatka': (1050, 60),
        'Ural': (800, 140),
        'Irkutsk': (950, 170),
        'Mongolia': (950, 260),
        'Japan': (1080, 260),
        'Afghanistan': (790, 280),
        'China': (900, 320),
        'Middle East': (700, 380),
        'India': (840, 400),
        'Siam': (960, 440),
        'Indonesia': (970, 570),
        'New Guinea': (1080, 540),
        'Western Australia': (1010, 700),
        'Eastern Australia': (1100, 650)
    }

    # Draw coordinates for verification
    if show_coords:
        for name, coords in territory_coords.items():
            draw.ellipse((coords[0]-5, coords[1]-5, coords[0]+5, coords[1]+5), fill=(255, 0, 0))

    # Draw troop numbers on the map
    for player in players_list:
        player_color = player_colors.get(player._name, (255, 255, 255))  # Default to white if not found
        print(f"Drawing troops for {player._name} with color {player_color}")  # Debug print
        for territory in player._territories:
            coords = territory_coords.get(territory._name, None)
            if coords and not show_coords:
                draw.text(coords, str(territory._troops), fill=player_color, font=font)

    # Save the annotated image
    image.save(output_image_path)




def run_simulation():
    ''' Run battle simulation.
    '''
    queue = initialize_playing_queue(players_list)
    place_troops(queue)
    save_game_state_to_image(players_list, output_image_path=1)
    while len(active_players_list) > 1:
        player = queue.popleft()
        add_troop_reinforcements(player)
        for i in range(1, 6):
            battles_dict = player.battles_dict()
            if battles_dict != {}:
                loser = battle(player)
                if determine_if_player_in_game(loser) == False:
                    active_players_list.remove(loser)
            else:
                break
        move_troops(player)
        queue.append(player)
    save_game_state_to_image(players_list, output_image_path=2)
    print(f'Winner is {active_players_list[0]._name}!')




if __name__ == "__main__":
    run_simulation()


