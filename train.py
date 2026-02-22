import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
import time
import sys
import argparse

# Import our custom modules
from model import SpellTransformer
from dataset_generation import get_or_create_dataset, get_tokenizer, SentenceDataset

# ==========================================
# 0. LOGGING SETUP (Captures all prints)
# ==========================================
class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8") # 'w' overwrites each run.

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect standard output
sys.stdout = Logger()

# ==========================================
# 1. ARGUMENT PARSING
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Gujarati Spell Checker Training")
    
    # Core Arguments
    parser.add_argument("--seed", type=int, default=None, 
                        help="Specific seed to use for training. If None, performs seed search.")
    
    # Hyperparameters (Optional overrides)
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning Rate")
    
    # Misc
    parser.add_argument("--log_file", type=str, default="log.txt", help="Log filename")
    
    # Model weights path (if resuming training)
    parser.add_argument("--model_path", type=str, default=None, help="Path to model weights to load")
    
    return parser.parse_args()

args = parse_args()

# ==========================================
# CONFIGURATION
# ==========================================
SEARCH_SEEDS = range(1, 51)       # Search seeds 1-50 (Only if args.seed is None)
FINAL_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
LIMIT_BATCHES_FOR_SEARCH = 256
MAX_SEQ_LEN = 128
VOCAB_SIZE = 6144

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Configuration: Epochs={FINAL_EPOCHS}, Batch={BATCH_SIZE}, LR={LR}")

# ==========================================
# 2. PREPARE DATA & TOKENIZER
# ==========================================
df = get_or_create_dataset()

# Train/Load Tokenizer
tokenizer = get_tokenizer(df['clean_correct'], vocab_size=VOCAB_SIZE)
PAD_IDX = tokenizer.token_to_id("<pad>")
SOS_IDX = tokenizer.token_to_id("<sos>")
EOS_IDX = tokenizer.token_to_id("<eos>")

# Create Dataset
full_dataset = SentenceDataset(df, tokenizer, MAX_SEQ_LEN)

# ==========================================
# 3. TRAINING UTILS
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_one_epoch(model, loader, optimizer, criterion, limit_batches=None):
    model.train()
    total_loss = 0
    count = 0
    
    for i, (x, y) in enumerate(loader):
        if limit_batches and i >= limit_batches: break
            
        x, y = x.to(device), y.to(device)
        tgt_input = y[:, :-1]
        tgt_output = y[:, 1:]

        optimizer.zero_grad()
        output = model(x, tgt_input)
        
        loss = criterion(
            output.reshape(-1, output.shape[-1]), 
            tgt_output.reshape(-1)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        count += 1
        
    return total_loss / count if count > 0 else float('inf')

# ==========================================
# 4. SEED SELECTION LOGIC
# ==========================================
if args.seed is not None:
    print(f"\n--- Skipping Search: Using provided seed {args.seed} ---")
    best_seed = args.seed
else:
    print(f"\n--- Starting Seed Search ({min(SEARCH_SEEDS)} to {max(SEARCH_SEEDS)}) ---")
    seed_results = {}

    for seed in SEARCH_SEEDS:
        set_seed(seed)
        g = torch.Generator()
        g.manual_seed(seed)
        loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g, num_workers=0)
        
        # Initialize Model
        model = SpellTransformer(
            vocab_size=tokenizer.get_vocab_size(),
            pad_idx=PAD_IDX,
            d_model=256,
            nhead=4,
            num_encoder_layers=3, 
            num_decoder_layers=2, 
            dim_feedforward=256,
            dropout=0.1
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
        
        # Run partial epoch
        avg_loss = train_one_epoch(model, loader, optimizer, criterion, limit_batches=LIMIT_BATCHES_FOR_SEARCH)
        print(f"Seed {seed:02d} | Loss: {avg_loss:.5f}")
        seed_results[seed] = avg_loss

    best_seed = min(seed_results, key=seed_results.get)
    print(f"\n🏆 WINNER: Seed {best_seed} (Loss: {seed_results[best_seed]:.5f})")

# ==========================================
# 5. FINAL FULL TRAINING
# ==========================================
print(f"\n--- Starting Full Training (Seed {best_seed}) ---")

set_seed(best_seed)
g = torch.Generator()
g.manual_seed(best_seed)
final_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g, num_workers=0)

final_model = SpellTransformer(
    vocab_size=tokenizer.get_vocab_size(),
    pad_idx=PAD_IDX,
    d_model=256, 
    nhead=4, 
    num_encoder_layers=3, 
    num_decoder_layers=2, 
    dim_feedforward=256,
    dropout=0.1
).to(device)

if args.model_path:
    print(f"Loading model weights from {args.model_path}...")
    final_model.load_state_dict(torch.load(args.model_path, map_location=device))

final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=LR, weight_decay=0.01)
final_criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
print(f"Total Trainable Parameters: {sum(p.numel() for p in final_model.parameters() if p.requires_grad)}")

best_tracked_loss = float('inf')

for epoch in range(FINAL_EPOCHS):
    start_time = time.time()
    epoch_loss = train_one_epoch(final_model, final_loader, final_optimizer, final_criterion)
    duration = time.time() - start_time
    
    print(f"Epoch {epoch+1}/{FINAL_EPOCHS} | Loss: {epoch_loss:.4f} | Time: {duration:.1f}s")
    
    if epoch_loss < best_tracked_loss:
        best_tracked_loss = epoch_loss
        torch.save(final_model.state_dict(), 'best_sentence_model.pt')
        print(f"  → Saved Best Model")
