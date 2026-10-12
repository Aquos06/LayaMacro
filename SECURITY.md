# Security policy

## Supported versions

LayaMacro is a research project with one maintained line: the latest commit on `main` and
the current checkpoint on [Hugging Face](https://huggingface.co/Aquos06/LayaMacro). Fixes
land there; older commits and checkpoints are not patched.

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Instead:

- [report it privately on GitHub](https://github.com/Aquos06/LayaMacro/security/advisories/new)
  (Security tab, *Report a vulnerability*), or
- email ruliciohansen@gmail.com with "LayaMacro security" in the subject.

Include what you found, the commit or checkpoint affected, the steps or a proof of concept
to reproduce it, and the impact you expect.

This is a small project maintained in spare time. We aim to acknowledge a report within
7 days and, once we understand the problem, to agree a disclosure date with you. We will
credit you in the fix unless you prefer not to be named.

## Scope

In scope: the code in this repository and the checkpoints it publishes. For example:

- loading a released checkpoint runs code, or reads or writes files it should not;
- a download or data script writes outside its target directory, or fetches from an
  unexpected source;
- a dependency locked in `uv.lock` has a known vulnerability that LayaMacro actually exercises.

Not security issues:

- **Wrong or poor predictions.** LayaMacro is not a medical device
  ([MODEL_LICENSE.md](MODEL_LICENSE.md)); please use the *Poor prediction* issue template.
- Vulnerabilities in upstream projects (PyTorch, transformers, `laya`, SigLIP 2,
  huggingface_hub). Report those to the project concerned, unless LayaMacro uses it in an
  unsafe way.

## Loading checkpoints safely

PyTorch checkpoints are pickle files, and fully unpickling one can run arbitrary code.
`load_predictor` loads `hf://` checkpoints with `torch.load(weights_only=True)`, which
accepts only tensors and plain values. A **local** file falls back to full unpickling when
needed, so that a training run's own checkpoint loads; the training and export scripts
unpickle their own caches and checkpoints the same way. Only load local `.pt` files that you
made or that come from a source you trust.
