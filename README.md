# Reproduction of the SipIt algorithm from "Language Models Are Injective and Hence Invertible"

[Paper on Arxiv](http://arxiv.org/abs/2510.15511)

Reproduction of the SipIt algorithm that allows to recover the token sequence from the hidden states of the LLM. This repo implements only the **Random Policy**, where the candidate token is selected randomly from the vocabulary.

## Setup

```bash
uv sync
```

## Run

```bash
uv run python sipit.py --sentence "A B C" --model_name "meta-llam
a/Llama-3.2-1B-Instruct" --layer_idx -1
```
