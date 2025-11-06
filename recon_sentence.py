import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sipit import sipit


def main(
    sentence: str,
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    layer_idx: int = -1,
    eps: float = 1e-3,
):
    """Recover token sequence from hidden states.

    Args:
        sentence: Sentence to recover
        model_name: Model name from HuggingFace
        layer_idx: Layer index to extract hidden states from
        eps: Acceptance threshold for embedding distance
    """
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    token_ids = tokenizer.encode(sentence, add_special_tokens=False)
    print(f"Token IDs to recover: {token_ids}")

    with torch.no_grad():
        outputs = model(torch.tensor([token_ids]), output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1][0]  # (n_tokens, d_model)

    recovered_token_ids = sipit(model, tokenizer, hidden_states, layer_idx, eps=eps)

    print(f"Recovered token IDs: {recovered_token_ids}")
    print(f"Recovered sentence: {tokenizer.decode(recovered_token_ids)}")


if __name__ == "__main__":
    fire.Fire(main)
