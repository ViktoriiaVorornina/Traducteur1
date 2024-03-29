from data import phrases

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

nlp_fr = spacy.load("fr_core_news_sm")

nlp_hu = huspacy.load("hu_core_news_lg")

SRC = Field(tokenize=lambda text: [tok.text for tok in nlp_fr.tokenizer(text)], init_token="<sos>", eos_token="<eos>", lower=True)
TRG = Field(tokenize=lambda text: [tok.text for tok in nlp_hu.tokenizer(text)], init_token="<sos>", eos_token="<eos>", lower=True)

fields = [('src', SRC), ('trg', TRG)]

train_size = int(0.8 * len(phrases))
train_examples = [Example.fromlist([fr, hu], fields) for fr, hu, in phrases[:train_size]]
test_examples = [Example.fromlist([fr, hu], fields) for fr, hu, in phrases[train_size:]]

train_data = Dataset(train_examples, fields)
test_data = Dataset(test_examples, fields)

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

BATCH_SIZE = 64
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data), 
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src)
)

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

        hidden = torch.zeros_like(trg_emb)

        trg_mask = nn.Transformer.generate_square_subsequent_mask(trg_seq_len).to(trg.device)
        print("Shape of trg_emb before multihead attention:", trg_emb.shape)
        output = self.decoder(trg_emb, encoder_outputs, tgt_mask=trg_mask)
        print("Shape of output after multihead attention:", output.shape)

        output = self.fc_out(output)

        return output


        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi[TRG.pad_token])
optimizer = optim.Adam(model.parameters(), lr=0.0001)

model = model.to(device)

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
        nlp = spacy.load("fr_core_news_sm")
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    
    with torch.no_grad():
        encoder_output = model.encoder(model.src_tok_emb(src_tensor)).permute(1, 0, 2)
        
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
                
        trg_mask = nn.Transformer.generate_square_subsequent_mask(len(trg_indexes)).to(device)
        
        with torch.no_grad():
            try:
                output = model.decoder(model.trg_tok_emb(trg_tensor), encoder_output, tgt_mask=trg_mask)
            except Exception as e:
                continue  # Пропускаем вывод сообщения об ошибке
        
        pred_token = output.argmax(2)[-1, :].item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:]


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

    example_idx = 0
    src = vars(test_data.examples[example_idx])['src']
    trg = vars(test_data.examples[example_idx])['trg']

    try:
        translation = translate_sentence(src, SRC, TRG, model, device)
        print(f'SRC: {" ".join(src)}')
        print(f'TRG: {" ".join(trg)}')
        print(f'PRED: {" ".join(translation)}')
    except AssertionError as e:
        print("An error occurred during translation:", e)
        continue
