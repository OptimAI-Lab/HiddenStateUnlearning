"""Fine-tune an unlearned checkpoint on the TOFU or MUSE forget set.

Evaluate saved checkpoints with eval.py (TOFU) or PARS/MUSE/eval.py (MUSE).
All model, tokenizer, data, and output paths are supplied at runtime.
"""

import argparse


def relearn(model_dir, data_file, out_dir, benchmark, tokenizer_dir=None,
            max_steps=25, per_device_batch_size=2, learning_rate=1e-5,
            max_len=4096, gradient_accumulation_steps=1,
            gradient_checkpointing=False, deepspeed=None,
            resume_from_checkpoint=None, seed=42):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    if __package__:
        from .dataset import RelearnDataset
    else:
        from dataset import RelearnDataset

    if min(max_steps, per_device_batch_size, gradient_accumulation_steps) < 1:
        raise ValueError("Training steps and batch sizes must be positive")
    set_seed(seed)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    # Initialize distributed training before loading the model for ZeRO-3.
    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        bf16=use_bf16,
        deepspeed=deepspeed,
        report_to="none",
        seed=seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir or model_dir)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer needs a padding or end-of-sequence token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dataset = RelearnDataset(data_file, tokenizer, benchmark, max_len=max_len)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
    )
    model.config.use_cache = False
    trainer = Trainer(model=model, train_dataset=dataset, args=training_args)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(out_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["TOFU", "MUSE"], required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--tokenizer_dir", default=None)
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=25)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    relearn(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
