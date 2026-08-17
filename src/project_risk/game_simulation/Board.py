


class Territory:
    
    def __init__(self, name, index, continent, neighbors):
        self._name = name
        self._index = index
        self._continent = continent
        self._neighbors  = neighbors
        
        self._owner = None
        self._troops = 0



''' Initialization of territories with name and continent association.
'''

# North America
alaska = Territory('Alaska', 1, 'North America', [])
northwest_territory = Territory('Northwest Territory', 2, 'North America', [])
greenland = Territory('Greenland', 3, 'North America', [])
alberta = Territory('Alberta', 4, 'North America', [])
ontario = Territory('Ontario', 5, 'North America', [])
quebec = Territory('Quebeck', 6, 'North America', [])
western_united_states = Territory('Western United States', 7, 'North America', [])
eastern_united_states = Territory('Eastern United States', 8, 'North America', [])
central_america = Territory('Central America', 9, 'North America', [])

# South America
venezuela = Territory('Venezuela', 10, 'South America', [])
brazil = Territory('Brazil', 11, 'South America', [])
peru = Territory('Peru', 12, 'South America', [])
argentina = Territory('Argentina', 13, 'South America', [])

# Africa 
north_africa = Territory('North Africa', 14, 'Africa', [])
egypt = Territory('Egypt', 15, 'Africa', [])
east_africa = Territory('East Africa', 16, 'Africa', [])
congo = Territory('Congo', 17, 'Africa', [])
south_africa = Territory('South Africa', 18, 'Africa', [])
madagaskar = Territory('Madagascar', 19, 'Africa', [])

# Europe
iceland = Territory('Iceland', 20, 'Europe', [])
scandinavia = Territory('Scandinavia', 21, 'Europe', [])
great_britain = Territory('Great Britain', 22, 'Europe', [])
northern_europe = Territory('Northern Europe', 23, 'Europe', [])
ukraine = Territory('Ukraine',24, 'Europe', [])
western_europe = Territory('Western Europe', 25, 'Europe', [])
southern_europe = Territory('Southern Europe', 26, 'Europe', [])

# Asia
siberia = Territory('Siberia', 27, 'Asia', [])
yakutsk = Territory('Yakutsk', 28, 'Asia', [])
kamchatka = Territory('Kamchatka', 29, 'Asia', [])
ural = Territory('Ural', 30, 'Asia', [])
irkutsk = Territory('Irkutsk', 31, 'Asia', [])
mongolia = Territory('Mongolia', 32, 'Asia', [])
japan = Territory('Japan', 33, 'Asia', [])
afghanistan = Territory('Afghanistan', 34, 'Asia', [])
china = Territory('China', 35, 'Asia', [])
middle_east = Territory('Middle East', 36, 'Asia', [])
india = Territory('India', 37, 'Asia', [])
siam = Territory('Siam', 38, 'Asia', [])

# Australia
indonesia = Territory('Indonesia', 39, 'Australia', [])
new_guinea = Territory('New Guinea', 40, 'Australia', [])
western_australia = Territory('Western Australia', 41, 'Australia', [])
eastern_australia = Territory('Eastern Australia', 42, 'Australia', [])


''' Assignment of neighbouring territories.
'''

alaska._neighbors = [northwest_territory, alberta, kamchatka]
northwest_territory._neighbors = [alaska, alberta, greenland, ontario]
alberta._neighbors = [alaska, northwest_territory, ontario, western_united_states]
greenland._neighbors = [northwest_territory, ontario, quebec, iceland]
ontario._neighbors = [greenland, northwest_territory, alberta, quebec, western_united_states, eastern_united_states]
quebec._neighbors = [greenland, ontario, eastern_united_states]
western_united_states._neighbors = [alberta, ontario, eastern_united_states, central_america]
eastern_united_states._neighbors = [central_america, western_united_states, ontario, quebec]
central_america._neighbors = [western_united_states, eastern_united_states, venezuela]
venezuela._neighbors = [central_america, brazil, peru]
brazil._neighbors = [venezuela, peru, argentina, north_africa]
peru._neighbors = [venezuela, brazil, argentina]
argentina._neighbors = [peru, brazil]
north_africa._neighbors = [brazil, western_europe, southern_europe, egypt, east_africa, congo]
egypt._neighbors = [north_africa, southern_europe, middle_east, east_africa]
east_africa._neighbors = [north_africa, egypt, middle_east, congo, south_africa, madagaskar]
congo._neighbors = [north_africa, east_africa, south_africa]
south_africa._neighbors = [congo, east_africa, madagaskar]
madagaskar._neighbors = [south_africa, east_africa]
iceland._neighbors = [greenland, great_britain, scandinavia]
scandinavia._neighbors = [iceland, great_britain, northern_europe, ukraine]
great_britain._neighbors = [iceland, scandinavia, northern_europe, western_europe]
northern_europe._neighbors = [great_britain, scandinavia, ukraine, southern_europe, western_europe]
ukraine._neighbors = [scandinavia, northern_europe, southern_europe, middle_east, afghanistan, ural]
western_europe._neighbors = [great_britain, northern_europe, southern_europe, north_africa]
southern_europe._neighbors = [western_europe, northern_europe, ukraine, middle_east, egypt, north_africa]
siberia._neighbors = [ural, china, yakutsk, mongolia, irkutsk]
yakutsk._neighbors = [siberia, irkutsk, kamchatka]
kamchatka._neighbors = [yakutsk, irkutsk, alaska, japan, mongolia]
ural._neighbors = [siberia, china, afghanistan, ukraine]
irkutsk._neighbors = [yakutsk, kamchatka, mongolia, siberia]
mongolia._neighbors = [china, siberia, irkutsk, kamchatka, japan]
japan._neighbors = [kamchatka, mongolia]
afghanistan._neighbors = [ural, china, india, middle_east, ukraine]
china._neighbors = [siam, india, afghanistan, ural, siberia, mongolia]
middle_east._neighbors = [afghanistan, india, east_africa, egypt, southern_europe, ukraine]
india._neighbors = [middle_east, afghanistan, china, siam]
siam._neighbors = [india, china, indonesia]
indonesia._neighbors = [siam, new_guinea, western_australia]
new_guinea._neighbors = [indonesia, eastern_australia]
western_australia._neighbors = [indonesia, eastern_australia]
eastern_australia._neighbors = [western_australia, new_guinea]


''' Definition of continents and member territories.
'''
continent_territory_dict = {'North America' : [alaska, northwest_territory, greenland, alberta, ontario, 
                                     quebec, western_united_states, eastern_united_states, 
                                     central_america],
                  'South America' : [venezuela, brazil, peru, argentina],
                  'Africa'        : [egypt, north_africa, east_africa, congo, south_africa, 
                                     madagaskar],
                  'Europe'        : [iceland, scandinavia, ukraine, great_britain, northern_europe, 
                                     western_europe, southern_europe],
                  'Asia'          : [siberia, yakutsk, kamchatka, ural, irkutsk, afghanistan, china, 
                                     japan, mongolia, middle_east, india, siam],
                  'Australia'     : [indonesia, new_guinea, western_australia, eastern_australia]}


continent_points_dict = {'North America' : 5,
                         'South America' : 2,
                         'Africa'        : 3,
                         'Europe'        : 5,
                         'Asia'          : 7,
                         'Australia'     : 2}




''' List of all territories.
'''
all_territories_list = [alaska, northwest_territory, greenland, alberta, ontario, quebec, 
                        western_united_states, eastern_united_states, central_america, venezuela, 
                        brazil, peru, argentina, egypt, north_africa, east_africa, congo, 
                        south_africa, madagaskar, iceland, scandinavia, ukraine, great_britain, 
                        northern_europe, western_europe, southern_europe, siberia, yakutsk, kamchatka, 
                        ural, irkutsk, afghanistan, china, japan, mongolia, middle_east, india, siam,
                        indonesia, new_guinea, western_australia, eastern_australia]


node_to_territory_dict = {territory._index: territory for territory in all_territories_list}




