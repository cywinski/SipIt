import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def accept(
    candidate_embedding: torch.Tensor,
    observed_embedding: torch.Tensor,
    eps: float = 1e-3,
) -> bool:
    """Check if candidate embedding matches observed embedding within tolerance.

    Args:
        candidate_embedding: Hidden state for candidate token. Shape: (d_model,)
        observed_embedding: Observed hidden state. Shape: (d_model,)
        eps: L2 distance tolerance.

    Returns:
        True if L2 distance <= eps, False otherwise.
    """
    distance = torch.linalg.norm(candidate_embedding - observed_embedding)

    return distance <= eps


def policy_random(vocab_size: int) -> list[int]:
    """Randomly select a candidate token ID.

    Args:
        vocab_size: Size of the vocabulary

    Returns:
        Candidate token ID
    """
    return torch.randperm(vocab_size).tolist()


def _gradient_policy_search(
    model: AutoModelForCausalLM,
    embedding_layer: torch.nn.Module,
    recovered_token_ids: list[int],
    observed_embedding: torch.Tensor,
    layer_idx: int,
    policy_steps: int,
    learning_rate: float,
    pbar: tqdm,
) -> list[int]:
    """
    Uses gradient descent to find the most likely candidate tokens.
    Returns a list of token IDs, ranked from most to least likely.
    """
    # Get prefix embeddings and all token embeddings from the embedding layer
    if len(recovered_token_ids) > 0:
        prefix_embeds = embedding_layer(
            torch.tensor([recovered_token_ids], device=model.device, dtype=torch.long)
        ).detach()
    else:
        prefix_embeds = torch.empty(
            1, 0, embedding_layer.weight.shape[1], device=model.device
        )
    all_token_embeddings = embedding_layer.weight.detach()

    # 1. Initialize a continuous "proxy" embedding that we can optimize
    proxy_embedding = all_token_embeddings.mean(0).clone().detach().to(model.device)
    proxy_embedding.requires_grad = True

    # 2. Setup optimizer for the proxy
    optimizer = torch.optim.Adam([proxy_embedding], lr=learning_rate)

    # 3. Perform gradient descent to make the model's output match the observed state
    for _ in range(policy_steps):
        optimizer.zero_grad()
        # Combine prefix with the current proxy to form the input
        inputs_embeds = torch.cat(
            [prefix_embeds, proxy_embedding.unsqueeze(0).unsqueeze(0)], dim=1
        )
        with torch.enable_grad():
            outputs = model(inputs_embeds=inputs_embeds, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer_idx]
        candidate_embedding = hidden_states[0, -1, :]

        loss = torch.linalg.norm(candidate_embedding - observed_embedding)
        pbar.set_description(f"Loss: {loss.item()}")
        loss.backward()
        optimizer.step()

    # 4. Rank all real vocabulary tokens by their L2 distance to the optimized proxy
    with torch.no_grad():
        distances = torch.linalg.norm(all_token_embeddings - proxy_embedding, dim=1)
        ranked_candidate_ids = torch.argsort(distances).tolist()

    return ranked_candidate_ids


def sipit(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    hidden_states: torch.Tensor,
    layer_idx: int = -1,
    eps: float = 1e-3,
    policy: str = "random",
    num_grad_steps: int = 10,
    lr: float = 0.1,
):
    """Recover token sequence by matching hidden states to embeddings (brute-force).

    Args:
        model: LLM
        tokenizer: Tokenizer
        hidden_states: Target hidden states. Shape: (n_tokens, d_model)
        layer_idx: Layer index to extract hidden states from
        eps: Acceptance threshold for embedding distance
        policy: Policy to use for selecting candidate tokens
        num_grad_steps: Number of gradient descent steps per candidate
        lr: Learning rate for gradient descent

    Returns:
        Recovered token IDs
    """
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    n_tokens, _ = hidden_states.shape
    recovered_token_ids: list[int] = []
    pbar = tqdm(total=n_tokens)
    embedding_layer = model.get_input_embeddings()
    for t in range(n_tokens):
        if policy == "random":
            candidate_indices: list[int] = policy_random(vocab_size)
        else:
            candidate_indices = _gradient_policy_search(
                model,
                embedding_layer=embedding_layer,
                recovered_token_ids=recovered_token_ids,
                observed_embedding=hidden_states[t].detach(),
                layer_idx=layer_idx,
                policy_steps=num_grad_steps,
                learning_rate=lr,
                pbar=pbar,
            )
        for j in range(vocab_size):
            pbar.set_description(
                f"Token {t + 1}/{n_tokens}: tested {(j / vocab_size * 100):.2f}% candidates"
            )
            current_sequence = recovered_token_ids + [candidate_indices[j]]
            with torch.no_grad():
                outputs = model(
                    torch.tensor(
                        [current_sequence], device=model.device, dtype=torch.long
                    ),
                    output_hidden_states=True,
                )
            current_hidden_states = outputs.hidden_states[layer_idx][0]
            candidate_embedding = current_hidden_states[-1]
            observed_embedding = hidden_states[t]
            if accept(observed_embedding, candidate_embedding, eps):
                # Hit!
                recovered_token_ids.append(candidate_indices[j])
                break
        pbar.update(1)
    pbar.close()
    return recovered_token_ids
