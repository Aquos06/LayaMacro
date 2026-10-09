# Laya: what it is, and how LayaMacro attaches an image to it

[Laya](https://huggingface.co/convaiinnovations/laya) (convaiinnovations, Apache-2.0) is an
encoder-only **typed decision model**: given a question, its options and some context text, it
returns a probability for each option in one forward pass, without generating any text.
LayaMacro starts from the `convaiinnovations/laya-typed-decisions` checkpoint
(ModernBERT-large backbone + a decision head, 421 M parameters).

Facts below were checked against the `laya` pip package (0.3.20) and by running it.

## Question types

| type | options | LayaMacro uses it for |
|---|---|---|
| `score` | ordered levels | the 5 nutrition questions: 18 ranges each |
| `choice` | named options | yes/no and "which of these foods" questions |
| `noul` | false / true | not used (see below) |

## Sequence and decision

`laya.common.build_sequence` writes:

```
[CLS] <type> question [SEP] [MASK] option 1 [MASK] option 2 ... [SEP] context [SEP]
```

Each option is preceded by its own `[MASK]` (id 50284), and the decision is read from the
hidden state **at each `[MASK]`'s absolute position** by a shared scorer, then a softmax over
the options.

## Five facts that shaped the graft

1. **Laya is not a `transformers` AutoModel.** It is a plain `nn.Module` in the `laya` pip
   package (`laya.load(...)`), so `AutoModel.from_pretrained` cannot load it.
2. **Its forward does not accept embeddings**, only token ids. The ModernBERT backbone does,
   so `layamacro.model.VisionLaya` re-implements the decision forward with an `inputs_embeds`
   path; a test checks it is identical to upstream when no image is given.
3. **Marker positions are absolute.** Tokens inserted *before* the options would shift every
   `[MASK]` and silently produce wrong answers. Image tokens therefore go into the context
   region, after the final option `[SEP]`: placeholder ids are reserved there by Laya's own
   `build_sequence`, and their embeddings are overwritten with the projected image tokens.
   `check_invariants` asserts on a batch that every marker lands on `[MASK]` and every slot
   holds exactly what the projector wrote.
4. **Don't pre-normalise the image tokens**: ModernBERT's embedding layer applies its own
   LayerNorm to whatever is injected. The projector instead matches the scale of Laya's
   token embeddings.
5. **Keep `sdpa` attention.** Flash-attention-2 unpads sequences, which breaks the 1:1
   position mapping the `[MASK]` gather relies on.

`laya.load()` checks checkpoints strictly, so a checkpoint carrying our extra modules cannot be
loaded by it. LayaMacro saves its own checkpoints and rebuilds the graft around a freshly
loaded Laya (`layamacro.graft.build_graft`).

## Known weaknesses, and how LayaMacro handles them

- **`noul` can follow its "false:/true:" option text instead of the input.** Yes/no questions
  are asked as a 2-option `choice` with neutral keys A/B in random order (upstream's own
  workaround).
- **`choice` with 11+ options ships with a broken temperature** (the checkpoint warns on
  load). LayaMacro's `choice` questions have at most 8 options, and nutrition answers use
  temperatures fitted on our calibration split.
- **Options share a token budget** (`head_max_len`), so long or many options get truncated.
  `build_image_sequence` raises instead of silently truncating options or image slots.
- **`score` is Laya's weakest question type upstream.** LayaMacro trains it with soft
  (HL-Gauss) targets over 18 bins.
- **Shipped probabilities are over-confident**; temperatures must be refitted. LayaMacro fits
  one temperature per target on a calibration split that is neither train nor test.
- **There is no Laya paper.** Cite the model card and the repository
  (https://github.com/NandhaKishorM/laya).
