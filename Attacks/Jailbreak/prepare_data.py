#!/usr/bin/env python3
"""
Prepare MUSE/TOFU forget data for GCG attack.

MUSE: Preserves VerbMem prompt/gt pairs as goal/target. Raw passage inputs
      instead use the first ~20 words as goal and the next ~50 as target.
TOFU: Uses forget QA pairs. Question = goal, Answer = target.

Output: CSV with columns 'goal' and 'target' for GCG attack.
"""

import argparse
import csv
import json
import os
import random


def split_passage_into_goal_target(passage, goal_words=20, target_words=50):
    """
    Split a passage into goal (first N words) and target (next M words).
    Used for MUSE raw text passages.
    """
    if goal_words < 1 or target_words < 1:
        raise ValueError('goal_words and target_words must be positive')
    words = passage.split()
    if len(words) < goal_words + 10:
        # Passage too short, use first half as goal, second half as target
        mid = len(words) // 2
        goal = " ".join(words[:mid])
        target = " ".join(words[mid:])
    else:
        goal = " ".join(words[:goal_words])
        target = " ".join(words[goal_words:goal_words + target_words])
    return goal.strip(), target.strip()


def prepare_muse(forget_json_path, output_csv, goal_words=20, target_words=50, max_samples=100):
    """
    Prepare MUSE forget data for GCG.
    Auto-detects format:
      - List of strings (raw passages): splits first N words as goal, next M as target
      - List of dicts (QA pairs): uses 'question' as goal, 'answer' as target
      - List of dicts (VerbMem): preserves 'prompt' as goal, 'gt' as target
    """
    with open(forget_json_path, 'r') as f:
        data = json.load(f)

    if max_samples < 1:
        raise ValueError('max_samples must be positive')
    goals, targets = [], []
    for item in data[:max_samples]:
        if isinstance(item, str):
            # Raw text passage
            passage = item.strip()
            if not passage:
                continue
            goal, target = split_passage_into_goal_target(passage, goal_words, target_words)
            if len(goal.split()) < 3 or len(target.split()) < 3:
                continue
        elif isinstance(item, dict):
            # QA or verbmem format
            goal = item.get('question', item.get('Question',
                   item.get('prompt', '')))
            target = item.get('answer', item.get('Answer',
                      item.get('gt', '')))
            if not goal or not target:
                continue
        else:
            continue
        goals.append(goal)
        targets.append(target)

    # Shuffle with fixed seed for deterministic train/test split
    combined = list(zip(goals, targets))
    random.Random(42).shuffle(combined)
    goals, targets = zip(*combined) if combined else ([], [])

    # Save as CSV
    if not goals:
        raise ValueError('No usable goal/target pairs found')
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['goal', 'target'])
        for g, t in zip(goals, targets):
            writer.writerow([g, t])
    print(f"MUSE: Saved {len(goals)} samples to {output_csv}")
    goal_lens = [len(g.split()) for g in goals]
    target_lens = [len(t.split()) for t in targets]
    print(f"  Goal avg length: {sum(goal_lens)/len(goal_lens):.0f} words")
    print(f"  Target avg length: {sum(target_lens)/len(target_lens):.0f} words")


def prepare_tofu(forget_json_path, output_csv, max_samples=100):
    """
    Prepare TOFU forget data for GCG.
    Input: JSON file with list of {'Question': ..., 'Answer': ...} dicts.
    """
    with open(forget_json_path, 'r') as f:
        qa_pairs = json.load(f)

    if max_samples < 1:
        raise ValueError('max_samples must be positive')
    goals, targets = [], []
    for item in qa_pairs[:max_samples]:
        # Handle both 'Question'/'Answer' and 'question'/'answer' keys
        question = item.get('Question', item.get('question', ''))
        answer = item.get('Answer', item.get('answer', ''))
        if not question or not answer:
            continue
        goals.append(question)
        targets.append(answer)

    # Shuffle with fixed seed for deterministic train/test split
    combined = list(zip(goals, targets))
    random.Random(42).shuffle(combined)
    goals, targets = zip(*combined) if combined else ([], [])

    if not goals:
        raise ValueError('No usable goal/target pairs found')
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['goal', 'target'])
        for g, t in zip(goals, targets):
            writer.writerow([g, t])
    print(f"TOFU: Saved {len(goals)} samples to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Prepare forget data for GCG jailbreak attack")
    parser.add_argument('--benchmark', choices=['MUSE', 'TOFU'], required=True)
    parser.add_argument('--forget_data', required=True, help="Path to forget JSON file")
    parser.add_argument('--output_csv', required=True, help="Output CSV path for GCG")
    parser.add_argument('--max_samples', type=int, default=100)
    parser.add_argument('--goal_words', type=int, default=20, help="MUSE: first N words as goal")
    parser.add_argument('--target_words', type=int, default=50, help="MUSE: next N words as target")

    args = parser.parse_args()

    if args.benchmark == 'MUSE':
        prepare_muse(args.forget_data, args.output_csv,
                     goal_words=args.goal_words,
                     target_words=args.target_words,
                     max_samples=args.max_samples)
    elif args.benchmark == 'TOFU':
        prepare_tofu(args.forget_data, args.output_csv,
                     max_samples=args.max_samples)


if __name__ == '__main__':
    main()
