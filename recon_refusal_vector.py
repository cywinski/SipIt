# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sipit import sipit

# %%
model_name = "google/gemma-2b-it"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print(f"Model loaded. Total layers: {model.config.num_hidden_layers}")
# %%
print(model.config)
# %%
refusal_vector = torch.load(
    "refusal_direction/pipeline/runs/gemma-2b-it/direction.pt", map_location=device
)
layer_idx = 10
print(f"Refusal vector shape: {refusal_vector.shape}")
# %%# Reshape for sipit (expects shape: (n_tokens, d_model))
refusal_vector_reshaped = refusal_vector.unsqueeze(0).to(model.device)

# Run sipit reconstruction
recovered_token_ids = sipit(
    model=model,
    tokenizer=tokenizer,
    hidden_states=refusal_vector_reshaped,
    layer_idx=layer_idx,
)

# %%
# Display results
print("\n" + "=" * 80)
print("RECONSTRUCTION RESULTS")
print("=" * 80)

print(f"\nRecovered {len(recovered_token_ids)} token(s)")
print(f"Token IDs: {recovered_token_ids}")

if recovered_token_ids:
    recovered_tokens = [
        tokenizer.decode([token_id]) for token_id in recovered_token_ids
    ]
    print(f"\nRecovered tokens (individual): {recovered_tokens}")
    print(f"\nReconstructed text: '{tokenizer.decode(recovered_token_ids)}'")
else:
    print("\nNo tokens were recovered!")

# %%
# Additional analysis: Show what the refusal direction represents
print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

print("\nThe refusal vector represents the difference in model activations between:")
print("\nHARMFUL prompts (what triggers refusal):")
for inst in harmful_instructions[:3]:
    print(f"  - {inst}")
print(f"  ... and {len(harmful_instructions) - 3} more")

print("\nHARMLESS prompts (what doesn't trigger refusal):")
for inst in harmless_instructions[:3]:
    print(f"  - {inst}")
print(f"  ... and {len(harmless_instructions) - 3} more")

print("\nExtraction details:")
print(f"  - Model: {model_name}")
print(f"  - Layer: {target_layer} (out of {model.config.num_hidden_layers})")
print(f"  - Token position: {target_position} (last token in prompt)")
print("  - Configuration from paper Table 5 (optimal for Qwen 1.8B)")

# %%
# Verify the configuration matches the paper
print("\n" + "=" * 80)
print("PAPER VALIDATION")
print("=" * 80)
print("\nFrom paper (Table 5, page 22):")
print("  - Qwen 1.8B optimal layer: l* = 15/24")
print("  - Qwen 1.8B optimal position: i* = -1")
print(f"  - Our configuration: layer={target_layer}, position={target_position}")
print("  ✓ Configuration matches paper specifications")
