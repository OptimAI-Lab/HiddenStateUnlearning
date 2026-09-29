# Do LLMs Really Forget? Hidden-State Leakage in Model Unlearning and How to Fix it

## Abstract

Unlearning in large language models (LLMs) is typically evaluated at the output level, where a model appears to suppress sensitive or undesirable content. In this work, we show that such evaluations can create an *illusion* of forgetting: even when output-level leakage is eliminated, sensitive information can remain encoded in the model’s hidden representations. We first provide a theoretical analysis establishing a fundamental separation between output suppression and representational erasure. Specifically, we show that the decoder can be made arbitrarily insensitive to sensitive directions, driving output-level leakage to zero, while the hidden representations retain the underlying information. To empirically validate this phenomenon, we train generative probe decoders on hidden states across transformer layers, enabling layer-wise measurement of information leakage. Across three widely used benchmarks, TOFU, MUSE, and WMDP, and state-of-the-art unlearning methods, we find that substantial sensitive information remains recoverable from hidden representations, even when standard output-level metrics indicate successful unlearning. To address this gap, we propose Probe-Adversarial Representation Suppression (PARS), an unlearning objective that adversarially minimizes the extractable information from hidden representations. PARS directly targets representational leakage and provides significantly stronger guarantees of erasure under adversarial probing and relearning attacks, outperforming all evaluated baselines. Our results highlight a fundamental limitation of existing unlearning paradigms and suggest that true forgetting in LLMs requires controlling not only model outputs, but also the information encoded in hidden representations.

## Getting Started

### Environment Setup

We use conda for environment management. Run the following commands from the repository root to set up the environment:

```bash
conda create -n unlearning python=3.11
conda activate unlearning
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install --no-build-isolation flash-attn==2.6.3
```

The commands above use CUDA 12.1; adjust the PyTorch installation for your CUDA version.

### How to Use

- **Measure hidden-state leakage:** [Probe](Probe/README.md) explains data preparation and probing for TOFU, MUSE, and WMDP.
- **Run PARS unlearning:** [PARS](PARS/README.md) provides data preparation and training commands for TOFU and MUSE.
- **Test unlearning robustness:** follow the [jailbreak](Attacks/Jailbreak/README.md) and [relearning](Attacks/Relearn/README.md) guides, or use [Leak@k](https://github.com/OptimAI-Lab/Leak-k).
