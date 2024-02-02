import random

#notre data
phrases = [
    ("Ő vezeti az autót.", "She drives the car.", "Elle conduit la voiture."),
    ("Ő főzi a vacsorát.", "He cooks dinner.", "Il cuisine le dîner."),
    ("Ő megjavítja a vízvezetéket.", "She fixes the plumbing.", "Elle répare la plomberie."),
    ("Ő vasalja a ruhákat.", "He irons the clothes.", "Il repasse les vêtements."),
    ("Ő épít egy házat.", "She builds a house.", "Elle construit une maison."),
    ("Ő varr egy ruhadarabot.", "He sews a garment.", "Il coud un vêtement."),
    ("Ő telepít polcokat.", "She installs shelves.", "Elle installe des étagères."),
    ("Ő festi a falakat.", "He paints the walls.", "Il peint les murs."),
    ("Ő nyírja a füvet.", "She mows the lawn.", "Elle tond la pelouse."),
    ("Ő megjavítja az autót.", "He fixes the car.", "Il répare la voiture."),
    ("Ő programoz egy szoftvert.", "She programs software.", "Elle programme un logiciel."),
    ("Ő díszíti a házat.", "He decorates the house.", "Il décore la maison."),
    ("Ő metszi a sövényeket.", "She trims the hedges.", "Elle taille les haies."),
    ("Ő tisztítja az ablakokat.", "He cleans the windows.", "Il nettoie les vitres."),
    ("Ő karbantartja a kertet.", "She maintains the garden.", "Elle entretient le jardin."),
    ("Ő locsolja a növényeket.", "He waters the plants.", "Il arrose les plantes."),
    ("Ő felújítja a konyhát.", "She renovates the kitchen.", "Elle rénove la cuisine."),
    ("Ő összeszereli a bútorokat.", "He assembles the furniture.", "Il assemble les meubles."),
    ("Ő pakolja a dobozokat.", "She moves the boxes.", "Elle déménage les cartons."),
    ("Ő mos ruhát.", "He does the laundry.", "Il fait la lessive."),
    ("Ő borotválja a szakállát.", "She shaves the beard.", "Elle tond la barbe."),
    ("Ő gyomlálja a kertet.", "He weeds the garden.", "Il désherbe le jardin."),
    ("Ő rajzol egy portrét.", "She draws a portrait.", "Elle dessine un portrait."),
    ("Ő farag egy szobrot.", "He sculpts a statue.", "Il sculpte une statue."),
    ("Ő fényképez egy tájat.", "She photographs the landscape.", "Elle photographie le paysage."),
    ("Ő ír egy dalt.", "He composes a song.", "Il compose une chanson."),
    ("Ő ír egy regényt.", "She writes a novel.", "Elle écrit un roman."),
    ("Ő rendez egy filmet.", "He directs a film.", "Il réalise un film."),
    ("Ő játszik egy hangszeren.", "She plays an instrument.", "Elle joue d'un instrument."),
    ("Ő zsonglőrködik labdákkal.", "He juggles balls.", "Il jongle avec des balles."),
    ("Ő zsonglőrködik gyertyatartókkal.", "She juggles clubs.", "Elle jongle avec des quilles."),
    ("Ő zsonglőrködik gyűrűkkel.", "He juggles rings.", "Il jongle avec des anneaux."),
    ("Ő egy kötélen jár.", "She walks on a tightrope.", "Elle marche sur un fil."),
    ("Ő akrobatikázik.", "He does acrobatics.", "Il fait des acrobaties."),
    ("Ő salsázik.", "She dances salsa.", "Elle danse la salsa."),
    ("Ő énekel egy dalt.", "He sings a song.", "Il chante une chanson."),
    ("Ő teniszezik.", "She plays tennis.", "Elle joue au tennis."),
    ("Ő focizik.", "He plays football.", "Il joue au football."),
    ("Ő úszik a medencében.", "She swims in the pool.", "Elle nage dans la piscine."),
    ("Ő fut a pályán.", "He runs on the track.", "Il court sur la piste."),
    ("Ő ugrál a kötélen.", "She jumps rope.", "Elle saute à la corde."),
    ("Ő dob egy diszkoszt.", "He throws a discus.", "Il lance un disque."),
    ("Ő ír egy regényt.", "She writes a novel.", "Elle écrit un roman."),
    ("Ő elkészíti a tortát.", "She bakes a cake.", "La pâtissière fait un gâteau."), 
    ("Ő összefogja a deszkákat.", "She nails the boards together.", "Elle cloue des planches ensemble."), 
    ("Ő hidratáló krémet használ.", "She uses a moisturising cream.", "Il utilise une crème hydratante."),
    ("Elle joue du piano.", "Zongorázik.", "She plays the piano."),
    ("Il fait du jardinage.", "Kertészkedik.", "He gardens."),
    ("Elle dessine un paysage.", "Tájképet rajzol.", "She draws a landscape."),
    ("Il écrit des poèmes.", "Verseket ír.", "He writes poems."),
    ("Elle cuisine des pâtisseries.", "Süteményeket süt.", "She bakes pastries."),
    ("Il sculpte des sculptures.", "Szobrokat farag.", "He carves sculptures."),
    ("Elle fait du bénévolat.", "Önkénteskedik.", "She volunteers."),
    ("Il pratique la méditation.", "Meditál.", "He meditates."),
    ("Elle peint des tableaux.", "Festményeket fest.", "She paints pictures."),
    ("Il fait du théâtre.", "Színházban játszik.", "He acts in the theater.")
]

random.shuffle(phrases) #melange des phrases

train_size = int(0.8 * len(phrases))
train_set = phrases[:train_size] #set pour étudier
test_set = phrases[train_size:] #set pour tester

# notre quantité de phrases dans chaque set
# print("Nombre de phrases dans le set pour étudier :", len(train_set))
# print("Nombre de phrases dans le set pour tester :", len(test_set))

import torchtext
from torchtext.data.utils import get_tokenizer

# Создаем токенизатор для каждого языка
tokenizer_fr = get_tokenizer('basic_english')
tokenizer_hu = get_tokenizer('basic_english')

# Функция токенизации для каждого языка
def tokenize_fr(text):
    return tokenizer_fr(text.lower())

def tokenize_hu(text):
    return tokenizer_hu(text.lower())

# Создаем объекты Field для каждого языка
SRC = torchtext.data.Field(tokenize=tokenize_fr, init_token='<sos>', eos_token='<eos>', lower=True)
TRG = torchtext.data.Field(tokenize=tokenize_hu, init_token='<sos>', eos_token='<eos>', lower=True)

# Создаем набор данных на основе списка фраз
fields = [('src', SRC), ('trg', TRG)]
train_examples = [torchtext.data.Example.fromlist([fr, hu], fields) for fr, hu, _ in train_set]

# Создаем объект Dataset из набора данных
train_data = torchtext.data.Dataset(train_examples, fields)

# Строим словари на основе обучающих данных
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

# Выводим размер словарей
print("Размер словаря для французского языка:", len(SRC.vocab))
print("Размер словаря для венгерского языка:", len(TRG.vocab))
