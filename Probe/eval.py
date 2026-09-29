from tqdm.contrib import tzip
from typing import List
from utils import EntailmentEvalLogger, get_prefix_before_words_occur

def eval_knowmem(
    model, tokenizer, questions: List[str], answers: List[str], max_new_tokens: int = 32
):
    # Map PyTorch device to an integer for the HuggingFace pipeline
    device_id = model.device.index if model.device.type == "cuda" else -1
    logger = EntailmentEvalLogger(device_id=device_id)

    for question, answer in tzip(questions, answers):
        prompt = f"Question: {question}\nAnswer: "
        input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=True).input_ids

        output_ids = model.generate(
            input_ids.to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        output_ids = output_ids[:, len(input_ids[0]):]
        output = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True,
        )[0]

        output = get_prefix_before_words_occur(output, ["\n\n", "\nQuestion", "Question:"])
        logger.log(prompt, answer, output, question=question)

    agg = logger.report()
    return agg, agg['logs']
