import os
import gc
import random
import shutil
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import read_data, write_csv, write_json
from train import train_decoder_early_stopping
from models import HybridWithTrainedDecoder, TrainableDecoder
from eval import eval_knowmem

def save_decoder(decoder: TrainableDecoder, save_path: str, layer: int, model_name: str = "original"):
    
    os.makedirs(save_path, exist_ok=True)
    decoder_path = os.path.join(save_path, f"decoder_{model_name}_layer_{layer}.pt")
    torch.save(decoder.state_dict(), decoder_path)
    return decoder_path

def load_decoder(decoder_path: str, target_model, device_id: int = 0):
    
    decoder = TrainableDecoder(target_model, device_id=device_id)
    decoder.load_state_dict(torch.load(decoder_path, map_location=f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"))
    decoder.eval()
    decoder.freeze()
    return decoder

def delete_model_from_cache(model_path):
    try:
        hf_cache_home = os.path.expanduser(os.getenv('HF_HOME', '~/.cache/huggingface'))
        models_dir = os.path.join(hf_cache_home, 'hub')
        
        if os.path.exists(models_dir):
            model_name_converted = f"models--{model_path.replace('/', '--')}"
            model_cache_path = os.path.join(models_dir, model_name_converted)
            
            if os.path.exists(model_cache_path):
                print(f"[*] Deleting model cache: {model_cache_path}")
                shutil.rmtree(model_cache_path)
                print(f"[✓] Successfully deleted model cache for {model_path}")
                return True
            else:
                print(f"[!] Model cache not found at {model_cache_path}")
                return False
    except Exception as e:
        print(f"[!] Error deleting model cache: {e}")
        return False

def stratified_split_by_author(qa_data, val_ratio=0.2, questions_per_author=20, seed=42):
    
    by_author = defaultdict(list)
    for idx, d in enumerate(qa_data):
        author_id = idx // questions_per_author
        by_author[author_id].append(d)

    counts = {a: len(v) for a, v in by_author.items()}
    unique_counts = set(counts.values())
    if len(unique_counts) != 1 or next(iter(unique_counts)) != questions_per_author:
        print(f"[!] WARNING: Uneven question counts per author: {counts}")
        print(f"[!] Expected {questions_per_author} per author. Author grouping may be wrong.")

    rng = random.Random(seed)
    train_data, val_data = [], []
    for author_id in sorted(by_author.keys()):
        items = by_author[author_id].copy()
        rng.shuffle(items)
        n_train = max(1, int(round(len(items) * (1 - val_ratio))))
        train_data.extend(items[:n_train])
        val_data.extend(items[n_train:])

    rng.shuffle(train_data)
    rng.shuffle(val_data)

    return train_data, val_data, len(by_author)

def process_model_sequential(model_name, model_path, train_qs, train_as, val_qs, val_as, 
                            tokenizer_dir, base_output_dir, logs_dir, num_layers, 
                            train_decoder_per_model=True, decoder_cache_dir=None, delete_after=True):
  
    gpu_id = 0
    print(f"[GPU {gpu_id}] Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    print(f"[GPU {gpu_id}] Loading Model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map={"": gpu_id}
    )
    model.eval()

    agg, base_logs = eval_knowmem(model, tokenizer, val_qs, val_as)
    baseline_score = agg['entailment_mean'] * 100
    
    write_json(base_logs, os.path.join(base_output_dir, f"{model_name.replace('/', '_')}_baseline_outputs.json"))
    print(f"[*] {model_name} Baseline ES: {baseline_score:.2f}%")

    for layer in range(num_layers):
        print(f"\n[GPU {gpu_id}] --- Processing {model_name} | Layer {layer} ---")
        
        if train_decoder_per_model:
            print(f"    [GPU {gpu_id}] -> Training decoder for {model_name} layer {layer}")
            decoder, info = train_decoder_early_stopping(
                target_model=model, tokenizer=tokenizer, 
                train_qs=train_qs, train_as=train_as,
                val_qs=val_qs, val_as=val_as,
                split_layer=layer, model_name=model_name, log_dir=logs_dir,
                device_id=gpu_id, # Target specific GPU
                max_epochs=50, batch_size=4, lr=5e-5, patience=3
            )
            
            if not info: 
                print(f"[GPU {gpu_id}] Skipping layer {layer} due to empty valid samples.")
                continue
                
            decoder.freeze()
        else:
            if decoder_cache_dir is None:
                print(f"[!] ERROR: decoder_cache_dir must be provided when train_decoder_per_model=False")
                continue
            
            decoder_path = os.path.join(decoder_cache_dir, f"decoder_original_layer_{layer}.pt")
            if not os.path.exists(decoder_path):
                print(f"[!] Decoder not found at {decoder_path}. Skipping layer {layer}.")
                continue
            
            decoder = load_decoder(decoder_path, model, device_id=gpu_id)
            info = {} 
        
        print(f"[GPU {gpu_id}] -> Evaluating Layer {layer} (Val Set)")
        hybrid_model = HybridWithTrainedDecoder(model, decoder, layer)
        
        agg, layer_logs = eval_knowmem(hybrid_model, tokenizer, val_qs, val_as)
        hybrid_score = agg['entailment_mean'] * 100
        print(f"[*] {model_name} Layer {layer} Decoder ES: {hybrid_score:.2f}%")
        write_json(layer_logs, os.path.join(base_output_dir, f"{model_name.replace('/', '_')}_layer_{layer}_outputs.json"))
        
        row = {
            'model_name': model_name,
            'layer': layer,
            'baseline_entailment': round(baseline_score, 2),
            'relearned_entailment': round(hybrid_score, 2),
        }
        
        if info:
            row['best_epoch'] = info['best_epoch']
            row['best_val_loss'] = round(info['best_val_loss'], 4)
        
        write_csv([row], os.path.join(base_output_dir, "detailed_layer_results.csv"), append=True)
        
        del hybrid_model
        del decoder
        torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[GPU {gpu_id}] Finished processing {model_name}.")

    if delete_after:
        print(f"[GPU {gpu_id}] Cleaning up disk space by deleting {model_name} from cache...")
        delete_model_from_cache(model_path)

def train_and_save_decoders_original(model_name, model_path, train_qs, train_as, val_qs, val_as, tokenizer_dir, base_output_dir, logs_dir, decoder_cache_dir, num_layers):
    gpu_id = 0
    print(f"[GPU {gpu_id}] Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    print(f"[GPU {gpu_id}] Loading Model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map={"": gpu_id})
    model.eval()

    for layer in range(num_layers):
        print(f"\n[GPU {gpu_id}] --- Training Decoder for Layer {layer} ---")
        
        decoder, info = train_decoder_early_stopping(
            target_model=model, tokenizer=tokenizer, 
            train_qs=train_qs, train_as=train_as,
            val_qs=val_qs, val_as=val_as,
            split_layer=layer, model_name=model_name, log_dir=logs_dir,
            device_id=gpu_id,
            max_epochs=50, batch_size=4, lr=5e-5, patience=3)
        
        if not info: 
            print(f"[GPU {gpu_id}] Skipping layer {layer} due to empty valid samples.")
            continue
        
        decoder.freeze()
        save_decoder(decoder, decoder_cache_dir, layer, model_name="original")
        del decoder
        torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()

def main():
    
    TOKENIZER_DIR = ...
    DATA_PATH = ...
    BASE_OUTPUT_DIR = ...
    TRAIN_DECODER_PER_MODEL = True
    LOGS_DIR = os.path.join(BASE_OUTPUT_DIR, "training_logs")
    DECODER_CACHE_DIR = os.path.join(BASE_OUTPUT_DIR, "decoder_cache") if not TRAIN_DECODER_PER_MODEL else None
    
    os.makedirs(LOGS_DIR, exist_ok=True)
    if DECODER_CACHE_DIR:
        os.makedirs(DECODER_CACHE_DIR, exist_ok=True)

    MODELS_TO_RUN = {
         "Original": ...,
         "NPO":      ...,
         "GradDiff": ...,
         "BLUR-NPO": ...,
         "SimNPO": ...,
         "RMU": ...,
         "PARS": ...,
    }
    
    NUM_LAYERS = 16
    VAL_RATIO = 0.2
    QUESTIONS_PER_AUTHOR = 20  
    qa_data = read_data(DATA_PATH)

    train_data, val_data, num_authors = stratified_split_by_author(
        qa_data,
        val_ratio=VAL_RATIO,
        questions_per_author=QUESTIONS_PER_AUTHOR,
        seed=42)

    n_train, n_val = len(train_data), len(val_data)
    train_qs = [d['question'] for d in train_data]
    train_as = [d['answer'] for d in train_data]
    val_qs = [d['question'] for d in val_data]
    val_as = [d['answer'] for d in val_data]

    print(f"[*] Stratified split by author: {num_authors} authors")
    print(f"[*]   Train: {n_train} samples ({n_train // num_authors} per author)")
    print(f"[*]   Val:   {n_val} samples ({n_val // num_authors} per author)")
    print(f"[*] Configuration: TRAIN_DECODER_PER_MODEL = {TRAIN_DECODER_PER_MODEL}")

    if not TRAIN_DECODER_PER_MODEL:
        original_model_name = None
        original_model_path = None

        for model_name, model_path in MODELS_TO_RUN.items():
            if "original" in model_name.lower() or model_name == "Original":
                original_model_name = model_name
                original_model_path = model_path
                break
        
        if original_model_name and original_model_path:
            print(f"\n{'='*70}\nTraining Decoders on Original Model: {original_model_name}\n{'='*70}")
            
            train_and_save_decoders_original(
                model_name=original_model_name,
                model_path=original_model_path,
                train_qs=train_qs,
                train_as=train_as,
                val_qs=val_qs,
                val_as=val_as,
                tokenizer_dir=TOKENIZER_DIR,
                base_output_dir=BASE_OUTPUT_DIR,
                logs_dir=LOGS_DIR,
                decoder_cache_dir=DECODER_CACHE_DIR,
                num_layers=NUM_LAYERS
            )
        else:
            print(f"[!] WARNING: No 'Original' model found in MODELS_TO_RUN. Cannot train decoders.")
            TRAIN_DECODER_PER_MODEL = True  

    
    for model_name, model_path in MODELS_TO_RUN.items():
        if not TRAIN_DECODER_PER_MODEL and (model_name.lower() == "original" or model_name == "Original"):
            print(f"\n[*] Skipping {model_name} (already used for decoder training)")
            continue
        
        print(f"\n{'='*70}\nStarting Sequential Evaluation: {model_name}\n{'='*70}")
        
        process_model_sequential(
            model_name=model_name,
            model_path=model_path,
            train_qs=train_qs,
            train_as=train_as,
            val_qs=val_qs,
            val_as=val_as,
            tokenizer_dir=TOKENIZER_DIR,
            base_output_dir=BASE_OUTPUT_DIR,
            logs_dir=LOGS_DIR,
            num_layers=NUM_LAYERS,
            train_decoder_per_model=TRAIN_DECODER_PER_MODEL,
            decoder_cache_dir=DECODER_CACHE_DIR,
            delete_after=True
        )

if __name__ == "__main__":
    main()