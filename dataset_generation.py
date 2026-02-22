import pandas as pd
import unicodedata
import re
import os
import random
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tokenizers import SentencePieceBPETokenizer, Tokenizer

SEED = 100
CSV_FILENAME = "processed_data.csv"
TOKENIZER_FILENAME = "gujarati_bpe_6k.json"

random.seed(SEED)

# ==========================================
# 1. ERROR GENERATION LOGIC
# ==========================================

# 1. HIGH-FREQUENCY PHONETIC CONFUSIONS
phonetic_confusion = {
    'શ': ['સ', 'ષ'],
    'સ': ['શ', 'ષ'],
    'ષ': ['શ', 'સ'],
    'ત': ['ટ'],
    'ટ': ['ત'],
    'દ': ['ડ'],
    'ડ': ['દ'],
    'ન': ['ણ'],
    'ણ': ['ન'],
    'ક': ['ખ'],
    'ખ': ['ક'],
    'ગ': ['ઘ'],
    'ઘ': ['ગ', 'ધ'],
    'ચ': ['છ'],
    'છ': ['ચ'],
    'જ': ['ઝ'],
    'ઝ': ['જ'],
    'ટ': ['ઠ', 'ત'],
    'ઠ': ['ટ'],
    'ડ': ['દ'],
    'ત': ['થ', 'ટ'],
    'થ': ['ત'],
    'દ': ['ધ', 'ડ'],
    'ધ': ['દ', 'ઘ'],
    'બ': ['ભ'],
    'ભ': ['બ'],
}

# 2. VOWEL MATRA CONFUSION
vowel_matra_confusion = {
    'િ': ['ી'],
    'ી': ['િ'],
    'ુ': ['ૂ'],
    'ૂ': ['ુ'],
    'ે': ['ૈ'],
    'ૈ': ['ે'],
    'ો': ['ૌ'],
    'ૌ': ['ો'],
    'ા': [''],
    '': ['ા'],
}

# 3. NASALIZATION
nasalization_confusion = {
    'ં': ['ઁ', 'ન'],
    'ઁ': ['ં'],
}

# 4. CONJUNCT/HALANT ERRORS
conjunct_errors = {
    'ક્ષ': ['ક્શ'],
    'જ્ઞ': ['ગ્ય', 'જ્ન'],
    'ત્ર': ['તર', 'ત્ત'],
    'શ્ર': ['સ્ર', 'શર'],
    'શ્ચ': ['સ્ચ', 'શચ'],
    'પ્ર': ['પર', 'પ્ય'],
    'સ્થ': ['સ્ત', 'સથ'],
    'ષ્ટ': ['ષ્ઠ', 'ષટ'],
    'સ્ત': ['સત'],
    'ન્ય': ['ન્ન', 'નય'],
    'ક્ત': ['ક્ય', 'કત'],
    'સ્ય': ['સ્થ', 'સય'],
    'દ્ધ': ['ધ્ધ', 'દધ'],
}

# 5. HALANT (VIRAMA) MISSING/EXTRA
halant_errors = [
    ('્ર', 'ર'),
    ('્ય', 'ય'),
    ('્વ', 'વ'),
    ('્ન', 'ન'),
    ('્મ', 'મ'),
    ('ર', '્ર'),
    ('ય', '્ય'),
]

# --- Error Functions ---
def phonetic_substitution_error(word, confusion_dict=phonetic_confusion):
    chars = list(word)
    valid_indices = [i for i, c in enumerate(chars) if c in confusion_dict]
    if not valid_indices: return word
    idx = random.choice(valid_indices)
    chars[idx] = random.choice(confusion_dict[chars[idx]])
    return "".join(chars)

def vowel_matra_error(word, confusion_dict=vowel_matra_confusion):
    chars = list(word)
    valid_indices = [i for i, c in enumerate(chars) if c in confusion_dict]
    if not valid_indices: return word
    idx = random.choice(valid_indices)
    chars[idx] = random.choice(confusion_dict[chars[idx]])
    return "".join(chars)

def conjunct_substitution_error(word, confusion_dict=conjunct_errors):
    possible_keys = [k for k in confusion_dict.keys() if k in word]
    if not possible_keys: return word
    targets = []
    for key in possible_keys:
        start = 0
        while True:
            idx = word.find(key, start)
            if idx == -1: break
            targets.append(key)
            break
    if not targets: return word
    target = random.choice(targets)
    return word.replace(target, random.choice(confusion_dict[target]), 1)

def halant_error(word, confusion_list=halant_errors):
    """Add or remove halant"""
    possible_matches = [pair for pair in confusion_list if pair[0] in word]
    
    if possible_matches:
        # Pick one specific rule randomly (e.g., replace '્ર' with 'ર')
        target, replacement = random.choice(possible_matches)
        return word.replace(target, replacement, 1)

    return word

def nasalization_error(word, confusion_dict=nasalization_confusion):
    chars = list(word)
    
    # 1. Check for Substitution Candidates
    # Find indices of characters that exist in our confusion dictionary
    valid_indices = [i for i, c in enumerate(chars) if c in confusion_dict]
    
    if valid_indices:
        # Pick a random spot to modify
        idx = random.choice(valid_indices)
        current_char = chars[idx]
        
        # Pick a replacement (e.g., 'ં' -> 'ન')
        replacement = random.choice(confusion_dict[current_char])
        chars[idx] = replacement
        return "".join(chars)
    
    # 2. Insertion (Noise)
    # If no nasal char found, randomly insert 'ં' to simulate "Extra Nasalization"
    if len(word) > 2:
        idx = random.randint(1, len(word)) # Random position
        return word[:idx] + 'ં' + word[idx:]
        
    return word

ERROR_TYPES = {
    'phonetic': (phonetic_substitution_error, phonetic_confusion),
    'vowel_matra': (vowel_matra_error, vowel_matra_confusion),
    'conjunct': (conjunct_substitution_error, conjunct_errors),
    'halant': (halant_error, None),
    'nasalization': (nasalization_error, nasalization_confusion),
}

# --- Sentence Logic ---
def apply_error_to_sentence(sentence, error_func, confusion_dict, num_errors):
    words = sentence.split()
    if not words: return sentence
    
    target_indices = list(range(len(words)))
    random.shuffle(target_indices)
    
    errors_applied = 0
    words_modified_mask = [False] * len(words)
    
    for idx in target_indices:
        if errors_applied >= num_errors: break
        if words_modified_mask[idx]: continue
            
        original_word = words[idx]
        if confusion_dict: new_word = error_func(original_word, confusion_dict)
        else: new_word = error_func(original_word)
            
        if new_word != original_word:
            words[idx] = new_word
            words_modified_mask[idx] = True
            errors_applied += 1
            
    return " ".join(words)

def generate_synthetic_sentence(correct_sentence, base_prob=0.5):
    variations = []
    variation_probs = [base_prob, base_prob + 0.1]
    num_variations = random.choice([1, 2])
    
    for i in range(num_variations):
        if num_variations <= 1:
            var_type = random.choice([0, 1])
            if random.random() > variation_probs[var_type]: continue
        
        curr_incorrect_sent = correct_sentence
        for error_name in ERROR_TYPES.keys():
            if num_variations <= 1:
                if random.random() > variation_probs[var_type]: continue
            else:
                if random.random() > variation_probs[i]: continue
            
            error_func, error_dict = ERROR_TYPES[error_name]
            count = random.choice([1, 2])
        
            incorrect_sent = apply_error_to_sentence(curr_incorrect_sent, error_func, error_dict, count)
            curr_incorrect_sent = incorrect_sent
        
        if curr_incorrect_sent != correct_sentence:
            variations.append(curr_incorrect_sent)
            
    return variations

# ==========================================
# 2. DATA PROCESSING PIPELINE
# ==========================================

def clean_text(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[\u0000-\u001F\u007F-\u009F\u200B-\u200F\u202A-\u202E\u2060\uFEFF]', ' ', text)
    text = re.sub(r'[\u0AE6-\u0AEF]', ' ', text)
    text = re.sub(r'[^\u0A80-\u0AFF\u0964\u0965\s]', ' ', text) # Strictly keep Gujarati + Dandas + Space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def has_long_word(text, max_len=25):
    if not isinstance(text, str): return False
    return any(len(word) > max_len for word in text.split())

def get_or_create_dataset():
    if os.path.exists(CSV_FILENAME):
        print(f"Loading existing dataset from {CSV_FILENAME}...")
        df = pd.read_csv(CSV_FILENAME)
        return df.dropna()
    
    print("Generating custom dataset (Original + Synthetic)...")
    dataset_hf = load_dataset("autopilot-ai/Gujarati-Grammarly-Datasets", split="train")
    df = pd.DataFrame(dataset_hf)[['Correct', 'Incorrect']].dropna()
    
    print("Cleaning text...")
    df['clean_correct'] = df['Correct'].apply(clean_text)
    df['clean_incorrect'] = df['Incorrect'].apply(clean_text)
    
    # --- FILTERS ---
    # 1. Drop empty
    df = df[(df['clean_incorrect'] != "") & (df['clean_correct'] != "")]
    
    # 2. Max Sentence Length (Characters) - Requirement: 128 chars
    df = df[df['clean_correct'].str.len() <= 128]
    df = df[df['clean_incorrect'].str.len() <= 128]
    
    # 3. Max Word Length - Requirement: No word > 25 chars
    # We invert the mask (keep rows where NO word is long)
    mask_long_correct = df['clean_correct'].apply(lambda x: has_long_word(x, 25))
    mask_long_incorrect = df['clean_incorrect'].apply(lambda x: has_long_word(x, 25))
    df = df[~(mask_long_correct | mask_long_incorrect)]
    
    print(f"Rows after filtering: {len(df)}")

    # --- SYNTHETIC GENERATION ---
    print("Generating synthetic variations...")
    synthetic_data = []
    unique_correct = df['clean_correct'].unique()
    print('Total unique correct sentences:', len(unique_correct))
    
    for sentence in unique_correct:
        variations = generate_synthetic_sentence(sentence)
        for var in variations:
            # We don't need to re-check word length here as clean_correct passed the check,
            # and our errors generally reduce length or swap chars, rarely increase beyond 25.
            synthetic_data.append({
                'clean_incorrect': var,
                'clean_correct': sentence
            })
            
    print(f"Original Pairs: {len(df)}")
    print(f"Synthetic Pairs Added: {len(synthetic_data)}")
    
    synthetic_df = pd.DataFrame(synthetic_data)
    final_df = pd.concat([df[['clean_incorrect', 'clean_correct']], synthetic_df])
    
    print(f"Saving {len(final_df)} pairs to {CSV_FILENAME}...")
    final_df.to_csv(CSV_FILENAME, index=False)
    
    return final_df

def get_tokenizer(df_correct_column, vocab_size=6144):
    if os.path.exists(TOKENIZER_FILENAME):
        print(f"Loading tokenizer from {TOKENIZER_FILENAME}...")
        return Tokenizer.from_file(TOKENIZER_FILENAME)
    
    print("Training Tokenizer...")
    tokenizer = SentencePieceBPETokenizer()
    tokenizer.train_from_iterator(
        df_correct_column,
        vocab_size=vocab_size,
        min_frequency=5,
        special_tokens=["<pad>", "<unk>", "<sos>", "<eos>"]
    )
    tokenizer.save(TOKENIZER_FILENAME)
    return tokenizer

class SentenceDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128): # Updated default to match config
        self.inputs = df['clean_incorrect'].values
        self.targets = df['clean_correct'].values
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sos_idx = tokenizer.token_to_id("<sos>")
        self.eos_idx = tokenizer.token_to_id("<eos>")
        self.pad_idx = tokenizer.token_to_id("<pad>")

    def __len__(self):
        return len(self.inputs)

    def encode(self, text):
        # Convert to string to avoid NaNs
        ids = self.tokenizer.encode(str(text)).ids
        ids = [self.sos_idx] + ids + [self.eos_idx]
        
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
            ids[-1] = self.eos_idx
        else:
            ids += [self.pad_idx] * (self.max_len - len(ids))
        return ids

    def __getitem__(self, idx):
        x = self.encode(self.inputs[idx])
        y = self.encode(self.targets[idx])
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)