# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import fire


def accept(
    candidate_embedding: torch.Tensor,
    observed_embedding: torch.Tensor,
    eps: float = 1e-3,
) -> bool:
    """Check if candidate embedding matches observed embedding within tolerance.

    Args:
        candidate_embedding: Hidden state for candidate token. Shape: (d_model,)
        observed_embedding: Observed hidden state. Shape: (d_model,)
        eps: L2 distance tolerance. Default: 1e-3.

    Returns:
        True if L2 distance <= eps, False otherwise.
    """
    distance = torch.linalg.norm(candidate_embedding - observed_embedding)

    return distance <= eps


def sipit(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    hidden_states: torch.Tensor,
    layer_idx: int = -1,
    eps: float = 1e-3,
):
    """Recover token sequence by matching hidden states to embeddings.

    Args:
        model: LLM
        tokenizer: Tokenizer
        hidden_states: Target hidden states. Shape: (n_tokens, d_model)
        layer_idx: Layer index to extract hidden states from
        eps: Acceptance threshold for embedding distance

    Returns:
        Recovered token IDs
    """
    vocab = tokenizer.get_vocab()
    n_tokens, _ = hidden_states.shape
    recovered_token_ids: list[int] = []
    for t in range(n_tokens):
        print(f"Recovering token {t + 1}/{n_tokens}...")
        tested_candidates: set[int] = set()
        for j in range(len(vocab)):
            if j in tested_candidates:
                continue
            print(f"Testing candidate {j}...")
            current_sequence = recovered_token_ids + [j]
            with torch.no_grad():
                outputs = model(
                    torch.tensor([current_sequence]), output_hidden_states=True
                )
            current_hidden_states = outputs.hidden_states[layer_idx][0]
            candidate_embedding = current_hidden_states[-1]
            observed_embedding = hidden_states[t]
            if accept(observed_embedding, candidate_embedding, eps):
                # Hit!
                recovered_token_ids.append(j)
                break
            else:
                tested_candidates.add(j)
    return recovered_token_ids


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
