# Backend Comparison Snapshot (2026-04-25)

Qualitative observations from manual comparison runs of the reference-image
+ split-composite path (PR #8). Sample basis: a recurring cyberpunk character
prompt with a 3x3 reference sheet input.

| Backend mode | Character fidelity | Background quality | Notable artifacts |
|---|---|---|---|
| `--image-backend gemini` | weak | clean composition | character traits drift from reference |
| `--image-backend openai` | high | grainy / noisy | reference-background bleed (e.g., katana traces) |
| `--image-backend openai --separate-character-bg` | high | clean | mild edge halo from chroma-key composite |

- `--separate-character-bg` produced the best end-to-end output in this
  comparison and is the recommended path for character-preservation work,
  despite the additional generation step (two OpenAI image calls + composite)
  and the modest edge-artifact tradeoff.
- The Gemini backend remains the default for prompts that prioritize scene
  composition over literal character preservation.
- This snapshot is qualitative (manual visual inspection), not a scored
  benchmark. C-group risk items (reverse grip, linear-object handling, soft-
  body deformation) remain on the deferred list.

This observation also closes the M3-deferred `gpt-image-2` forbidden-effect
evaluation: forbidden constraints were respected in the OpenAI runs, but
reference-background bleed dominated other concerns and motivated the
split-composite route.
