import torch
import torch.nn.functional as F
import warnings
import argparse
import time
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

# Import our custom modules
from model import SpellTransformer
from dataset_generation import get_tokenizer, get_or_create_dataset

warnings.filterwarnings("ignore")

# ==========================================
# 1. ARGUMENT PARSING
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Gujarati Spell Checker Inference")
    
    parser.add_argument("--model_path", type=str, default="best_sentence_model.pt", 
                        help="Path to the trained model file")
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="Inference batch size (Only used for Greedy Search)")
    parser.add_argument("--samples", type=int, default=4096, 
                        help="Number of test samples to run")
    parser.add_argument("--beam_width", type=int, default=3, 
                        help="Beam width. 1 = Fast Greedy (Batch), >1 = Beam Search (Slower, Accurate)")
    parser.add_argument("--output_file", type=str, default="test_results.txt", 
                        help="File to save the results")
    
    return parser.parse_args()

args = parse_args()

# ==========================================
# 2. CONFIGURATION
# ==========================================
MAX_SEQ_LEN = 128
VOCAB_SIZE = 6144
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"Samples: {args.samples}")
print(f"Beam Width: {args.beam_width} ({'Batch Greedy' if args.beam_width==1 else 'Beam Search'})")

# ==========================================
# 3. LOAD RESOURCES
# ==========================================
print("Loading dataset & tokenizer...")
df = get_or_create_dataset()

# Load Tokenizer
tokenizer = get_tokenizer(df['clean_correct'], vocab_size=VOCAB_SIZE)
PAD_IDX = tokenizer.token_to_id("<pad>")
SOS_IDX = tokenizer.token_to_id("<sos>")
EOS_IDX = tokenizer.token_to_id("<eos>")

# Initialize Model
print("Loading model...")
model = SpellTransformer(
    vocab_size=tokenizer.get_vocab_size(),
    pad_idx=PAD_IDX,
    d_model=256, 
    nhead=4, 
    num_encoder_layers=3, 
    num_decoder_layers=2, 
    dim_feedforward=256,
    dropout=0.1
).to(DEVICE)

model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
model.eval()

# ==========================================
# 4. INFERENCE METHODS
# ==========================================

def correct_sentence_batch_greedy(model, tokenizer, sentences, device, max_len=128):
    """
    Fast processing for Beam Width = 1
    """
    model.eval()
    batch_size = len(sentences)
    
    # Tokenize Batch
    src_tensors = []
    for s in sentences:
        ids = tokenizer.encode(s).ids
        ids = [SOS_IDX] + ids + [EOS_IDX]
        if len(ids) > max_len: ids = ids[:max_len]
        src_tensors.append(torch.tensor(ids, dtype=torch.long))
        
    src = pad_sequence(src_tensors, batch_first=True, padding_value=PAD_IDX).to(device)
    tgt = torch.full((batch_size, 1), SOS_IDX, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        for _ in range(max_len):
            output = model(src, tgt)
            next_token_logits = output[:, -1, :]
            next_tokens = next_token_logits.argmax(dim=-1).unsqueeze(1)
            
            is_eos = (next_tokens.squeeze(1) == EOS_IDX)
            finished = finished | is_eos
            tgt = torch.cat([tgt, next_tokens], dim=1)
            
            if finished.all():
                break
                
    decoded_sentences = []
    for seq in tgt.tolist():
        valid_ids = []
        for token_id in seq:
            if token_id == SOS_IDX: continue
            if token_id == EOS_IDX: break
            valid_ids.append(token_id)
        decoded_sentences.append(tokenizer.decode(valid_ids))
        
    return decoded_sentences

def correct_sentence_beam(model, tokenizer, sentence, device, beam_width=3, max_len=128):
    """
    High Accuracy processing for Beam Width > 1
    Processed one by one (Slower but accurate)
    """
    model.eval()
    src_ids = tokenizer.encode(sentence).ids
    src_ids = [SOS_IDX] + src_ids + [EOS_IDX]
    src = torch.tensor(src_ids).unsqueeze(0).to(device)
    
    # (log_prob, sequence_list)
    candidates = [(0.0, [SOS_IDX])]
    
    with torch.no_grad():
        for _ in range(max_len):
            new_candidates = []
            
            for score, seq in candidates:
                if seq[-1] == EOS_IDX:
                    new_candidates.append((score, seq))
                    continue
                
                tgt = torch.tensor(seq).unsqueeze(0).to(device)
                output = model(src, tgt)
                
                # Get log probabilities
                next_token_logits = output[0, -1, :]
                log_probs = F.log_softmax(next_token_logits, dim=-1)
                
                # Expand Top-K
                topk_probs, topk_ids = torch.topk(log_probs, beam_width)
                
                for i in range(beam_width):
                    token_id = topk_ids[i].item()
                    token_score = topk_probs[i].item()
                    new_candidates.append((score + token_score, seq + [token_id]))
            
            # Keep top K best sequences
            candidates = sorted(new_candidates, key=lambda x: x[0], reverse=True)[:beam_width]
            
            # Stop if all finished
            if all(c[1][-1] == EOS_IDX for c in candidates):
                break
                
    # Decode best candidate
    best_seq = candidates[0][1]
    valid_ids = [idx for idx in best_seq if idx not in [SOS_IDX, EOS_IDX, PAD_IDX]]
    return tokenizer.decode(valid_ids)

# ==========================================
# 5. EXECUTION LOOP
# ==========================================
print("\n--- Starting Inference ---")

test_df = df.sample(n=args.samples, random_state=42)
all_incorrect = test_df['clean_incorrect'].tolist()
all_correct = test_df['clean_correct'].tolist()

results = []
correct_count = 0
total_processed = 0
start_time = time.time()

# LOGIC SWITCH: BATCH vs SINGLE BEAM
if args.beam_width == 1:
    # --- Fast Batch Greedy ---
    for i in tqdm(range(0, len(all_incorrect), args.batch_size), desc="Batch Processing"):
        batch_incorrect = all_incorrect[i : i + args.batch_size]
        batch_correct = all_correct[i : i + args.batch_size]
        
        predictions = correct_sentence_batch_greedy(model, tokenizer, batch_incorrect, DEVICE, MAX_SEQ_LEN)
        
        for pred, truth, original in zip(predictions, batch_correct, batch_incorrect):
            is_correct = (pred.strip() == truth.strip())
            if is_correct: correct_count += 1
            results.append(f"{original} - {truth} - {pred} - {is_correct}")
        
        total_processed += len(batch_incorrect)

else:
    # --- Accurate Beam Search ---
    print(f"Running Beam Search (Width={args.beam_width})... This will take longer.")
    for original, truth in tqdm(zip(all_incorrect, all_correct), total=len(all_incorrect), desc="Beam Processing"):
        
        pred = correct_sentence_beam(model, tokenizer, original, DEVICE, args.beam_width, MAX_SEQ_LEN)
        
        is_correct = (pred.strip() == truth.strip())
        if is_correct: correct_count += 1
        results.append(f"{original} - {truth} - {pred} - {is_correct}")
        
        total_processed += 1

duration = time.time() - start_time
accuracy = (correct_count / total_processed) * 100

# ==========================================
# 6. SAVE RESULTS
# ==========================================
print(f"\nFinished processing {total_processed} samples.")
print(f"Time taken: {duration:.2f}s ({total_processed/duration:.1f} samples/sec)")
print(f"Accuracy: {accuracy:.2f}%")

with open(args.output_file, "w", encoding="utf-8") as f:
    f.write(f"Mode: {'Beam Search' if args.beam_width > 1 else 'Greedy Batch'}\n")
    f.write(f"Beam Width: {args.beam_width}\n")
    f.write(f"Accuracy: {accuracy:.2f}%\n")
    f.write(f"Total Samples: {total_processed}\n")
    f.write("="*40 + "\n")
    f.write("\n".join(results))

print(f"Results saved to {args.output_file}")