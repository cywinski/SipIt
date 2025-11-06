# Reproduction of the SipIt algorithm from "Language Models Are Injective and Hence Invertible"

[Paper on Arxiv](http://arxiv.org/abs/2510.15511)

Reproduction of the SipIt algorithm that allows to recover the token sequence from the hidden states of the LLM. This repo implements both the **Random Policy** and the **Gradient-based Policy**.

## Setup

```bash
uv sync
```

Set your HuggingFace token in the `.env` file.

## Run

### Random Policy
Candidate tokens are selected randomly from the vocabulary.

```bash
uv run python recon_sentence.py --sentence "A B Z" --model_name "meta-llama/Llama-3.2-1B-Instruct" --layer_idx -1 --eps 1e-3 --policy "random"
```

### Gradient-based Policy
Candidate tokens are sorted by their distance to the target hidden state using gradient descent.

```bash
uv run python recon_sentence.py --sentence "A B Z" --model_name "meta-llama/Llama-3.2-1B-Instruct" --layer_idx -1 --eps 1e-3 --policy "gradient" --num_grad_steps 300 --lr 0.5
```
