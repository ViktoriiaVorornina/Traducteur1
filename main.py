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


import torchtext
from torchtext.data import Field, Example, Dataset, BucketIterator
from torchtext.data.utils import get_tokenizer
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.nn import Transformer
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import spacy
import huspacy


# Загрузка моделей SpaCy для французского и английского языков
nlp_fr = spacy.load("fr_core_news_sm")
nlp_en = spacy.load("en_core_web_sm")


# Определение поля для токенизации
SRC = Field(tokenize=lambda text: [tok.text for tok in nlp_fr.tokenizer(text)], init_token="<sos>", eos_token="<eos>", lower=True)
TRG = Field(tokenize=lambda text: [tok.text for tok in nlp_en.tokenizer(text)], init_token="<sos>", eos_token="<eos>", lower=True)
SRC = Field(tokenize=lambda text: [tok.text for tok in nlp_en.tokenizer(text)], init_token="<sos>", eos_token="<eos>", lower=True)

# Определение полей данных
fields = [('src', SRC), ('trg', TRG)]

# Создание примеров и датасета
train_size = int(0.8 * len(phrases))
train_examples = [Example.fromlist([fr, hu], fields) for fr, hu, _ in phrases[:train_size]]
test_examples = [Example.fromlist([fr, hu], fields) for fr, hu, _ in phrases[train_size:]]

train_data = Dataset(train_examples, fields)
test_data = Dataset(test_examples, fields)

# Построение словарей
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

# Определение итераторов
BATCH_SIZE = 64
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data), 
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src)
)

# Определение параметров модели
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
EMB_DIM = 256
N_LAYERS = 4
N_HEADS = 8
FFN_HID_DIM = 512
N_ENCODER_LAYERS = 3
N_DECODER_LAYERS = 3
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

# Создание модели
class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        emb_dim,
        n_encoder_layers,
        n_decoder_layers,
        n_heads,
        encoder_dim_ff,
        decoder_dim_ff,
        encoder_dropout,
        decoder_dropout,
    ):
        super().__init__()
        
        encoder_layer = TransformerEncoderLayer(emb_dim, n_heads, encoder_dim_ff, encoder_dropout)
        decoder_layer = TransformerDecoderLayer(emb_dim, n_heads, decoder_dim_ff, decoder_dropout)
        
        self.encoder = TransformerEncoder(encoder_layer, n_encoder_layers)
        self.decoder = TransformerDecoder(decoder_layer, n_decoder_layers)
        
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_dim)
        self.trg_tok_emb = nn.Embedding(trg_vocab_size, emb_dim)
        
        self.fc_out = nn.Linear(emb_dim, trg_vocab_size)
        
        self._reset_parameters()
        
    def forward(self, src, trg):
        src_seq_len, N = src.size()
        trg_seq_len, N = trg.size()
        
        src_emb = self.src_tok_emb(src)
        trg_emb = self.trg_tok_emb(trg)
        
        src_emb = src_emb * (EMB_DIM ** 0.5)
        trg_emb = trg_emb * (EMB_DIM ** 0.5)
        
        encoder_outputs = self.encoder(src_emb)
        
        # Инициализируем начальное скрытое состояние декодера нулевым тензором
        hidden = torch.zeros_like(trg_emb)
        
        # Маска для автономного предсказания
        trg_mask = nn.Transformer.generate_square_subsequent_mask(trg_seq_len).to(trg.device)
        
        # Получаем выходы декодера
        output = self.decoder(trg_emb, encoder_outputs, tgt_mask=trg_mask)
        
        # Преобразуем выход декодера
        output = self.fc_out(output)
        
        return output

        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

# Инициализация модели
model = Seq2SeqTransformer(
    INPUT_DIM,
    OUTPUT_DIM,
    EMB_DIM,
    N_ENCODER_LAYERS,
    N_DECODER_LAYERS,
    N_HEADS,
    FFN_HID_DIM,
    FFN_HID_DIM,
    ENC_DROPOUT,
    DEC_DROPOUT,
)

# Инициализация устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi[TRG.pad_token])
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Перемещение модели на устройство
model = model.to(device)

# Функция обучения модели
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

# Функция оценки модели
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg)
            
            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
    model.eval()
    if isinstance(sentence, str):
        nlp = spacy.load("en_core_web_sm")
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    
    with torch.no_grad():
        encoder_outputs = model.encoder(model.src_tok_emb(src_tensor)).permute(1, 0, 2)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
                
        with torch.no_grad():
            output = model.decoder(model.trg_tok_emb(trg_tensor), encoder_outputs)
        
        pred_token = output.argmax(2)[-1, :].item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:]


# Обучение модели
N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, test_iterator, criterion)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model.pt')
    
    print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

# Оценка модели на тестовом наборе данных
example_idx = 0
src = vars(test_data.examples[example_idx])['src']
trg = vars(test_data.examples[example_idx])['trg']

translation = translate_sentence(src, SRC, TRG, model, device)
print(f'SRC: {" ".join(src)}')
print(f'TRG: {" ".join(trg)}')
print(f'PRED: {" ".join(translation)}')