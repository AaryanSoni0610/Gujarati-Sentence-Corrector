import torch
import torch.nn as nn
import math

class SpellTransformer(nn.Module):
    def __init__(self, vocab_size, pad_idx, d_model=256, nhead=4, 
                 num_encoder_layers=3, num_decoder_layers=2, 
                 dim_feedforward=256, dropout=0.1):
        super().__init__()
        
        self.pad_idx = pad_idx
        self.d_model = d_model
        
        # 1. Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 2. Transformer Core
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # 3. Output Head
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.fc_out.weight = self.embedding.weight

    def forward(self, src, tgt):
        # Create Masks
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device).bool()
        
        # Padding Masks
        src_pad_mask = (src == self.pad_idx)
        tgt_pad_mask = (tgt == self.pad_idx)

        # Apply Embeddings + Positional Encoding
        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        # Transformer Pass
        out = self.transformer(
            src=src, 
            tgt=tgt, 
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )

        return self.fc_out(out) 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div)
        pe[:, 0, 1::2] = torch.cos(position * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)