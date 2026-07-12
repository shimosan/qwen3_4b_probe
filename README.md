# qwen3_4b_probe

**Looking inside a small LLM — observing and visualizing the internals of Qwen3-4B.**

A teaching repository that probes Qwen3-4B's internal computation — hidden states, attention, logits, the residual stream, and the token-embedding geometry — through actual tensors and figures, using stock Hugging Face Transformers APIs (no source modification).

🇯🇵 日本語で読む → **[詳しい日本語ガイド `guide_ja.md`](guide_ja.md)**（セットアップ・全ノート一覧・実験レポート）

---

[![Qwen3-4B multilingual hierarchical clustering: English is bundled with Chinese/Japanese/Korean](images/mling_demo_dendro_ward.png)](multilingual_geometry.md)

**Figure 1**: Hierarchical clustering of 38 languages measured from Qwen3-4B's token embedding matrix $W_E$ (Ward linkage; distance = √(1 − mean cosine similarity)). **English leaves its own Germanic family and joins the Chinese / Japanese / Korean cluster, with English–Chinese the closest pair**. This is consistent with the training-data mix rather than linguistic genealogy (the English-pivot dictionary is a confound). The demo also estimates per-language orthogonal maps $R(L)$ and, comparing three accounts of $W_E$ (concept-first / language-first / additive), finds the concept-first model $v_L(w)=R(L)R(w)v_o$ best supported; removing the concept component then makes the hidden language structure surface.

Read the illustrated write-up: **Multilingual geometry encoded in Qwen3-4B's token embedding matrix** → [English](multilingual_geometry_en.md) · [日本語 (Japanese)](multilingual_geometry.md)

![Qwen3-4B Logit Lens for "The capital of Japan is"](images/nb02_logit_lens_grid_clean.png)

**Figure 2**: Logit Lens on the prompt "The capital of Japan is" (from notebook [02 residual_stream / logit_lens / patching](lecture/02_residual_stream_logit_lens_patching.ipynb)). Each layer's residual stream (vertical: bottom = embedding, top = final layer) is projected through the output embedding to the vocabulary; each cell shows the top predicted token at that position (horizontal: input token → next token), colored by the gold token's rank (log scale, yellow = rank 1). "is → Tokyo" rises to rank 1 around layer 30. Notably, in the "Japan → is" column the middle layers surface **Chinese** tokens for Japan / Tokyo (e.g. simplified 东京, distinct from Japanese 東京) before converging to the surface form. Internally the model appears to route the Japan concept through Chinese.

---

## Notebooks

Each notebook is **self-contained** and runs on the `aidemo2026` venv (dependencies pinned in [`requirements.txt`](requirements.txt)). **Open in Colab** to run in the browser with zero setup — for GPU notebooks, switch the runtime type to a GPU.

| Notebook | What it covers | Run |
|---|---|---|
| **[00 intro_chat](lecture/00_intro_chat.ipynb)** | Model loading, tokenizer, chat template, greedy decoding | [rendered](rendered/00_intro_chat.ipynb) · [![nbviewer](https://img.shields.io/badge/Render-nbviewer-orange)](https://nbviewer.org/github/shimosan/qwen3_4b_probe/blob/main/rendered/00_intro_chat.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shimosan/qwen3_4b_probe/blob/main/lecture/00_intro_chat.ipynb) |
| **[01 tokenizer](lecture/01_tokenizer.ipynb)** | Unicode / UTF-8, encode / decode, special tokens | [rendered](rendered/01_tokenizer.ipynb) · [![nbviewer](https://img.shields.io/badge/Render-nbviewer-orange)](https://nbviewer.org/github/shimosan/qwen3_4b_probe/blob/main/rendered/01_tokenizer.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shimosan/qwen3_4b_probe/blob/main/lecture/01_tokenizer.ipynb) |
| **[02 residual_stream / logit_lens / patching](lecture/02_residual_stream_logit_lens_patching.ipynb)** | Residual stream, Logit Lens, Activation Patching (main notebook) | [rendered](rendered/02_residual_stream_logit_lens_patching.ipynb) · [![nbviewer](https://img.shields.io/badge/Render-nbviewer-orange)](https://nbviewer.org/github/shimosan/qwen3_4b_probe/blob/main/rendered/02_residual_stream_logit_lens_patching.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shimosan/qwen3_4b_probe/blob/main/lecture/02_residual_stream_logit_lens_patching.ipynb) |
| **[multilingual_geometry_demo](lecture/multilingual_geometry_demo.ipynb)** ([EN](lecture/multilingual_geometry_demo_en.ipynb)) | Multilingual geometry of $W_E$ (figure above; rotations $R(L)$) | [rendered](rendered/multilingual_geometry_demo.ipynb) · [![nbviewer](https://img.shields.io/badge/Render-nbviewer-orange)](https://nbviewer.org/github/shimosan/qwen3_4b_probe/blob/main/rendered/multilingual_geometry_demo.ipynb) · [rendered (EN)](rendered/multilingual_geometry_demo_en.ipynb) · [![nbviewer (EN)](https://img.shields.io/badge/Render-nbviewer%20(EN)-orange)](https://nbviewer.org/github/shimosan/qwen3_4b_probe/blob/main/rendered/multilingual_geometry_demo_en.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shimosan/qwen3_4b_probe/blob/main/lecture/multilingual_geometry_demo.ipynb) |
| **[wordvec_demo](lecture/wordvec_demo.ipynb)** | Word-vector primer with GloVe (standalone; not Qwen-specific) | [rendered](rendered/wordvec_demo.ipynb) · [![nbviewer](https://img.shields.io/badge/Render-nbviewer-orange)](https://nbviewer.org/github/shimosan/qwen3_4b_probe/blob/main/rendered/wordvec_demo.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shimosan/qwen3_4b_probe/blob/main/lecture/wordvec_demo.ipynb) |

---

## Model

Primary target [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) (comparisons: [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B), [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B)). Model weights live in the Hugging Face cache and are never stored in the repository.

## License / Acknowledgements

Released under the MIT License ([LICENSE](LICENSE)). This work builds on Qwen3, Transformers, TransformerLens, mwhanna's transcoders, Qwen-Scope, GloVe, and gensim — see the full acknowledgements in [guide_ja.md](guide_ja.md#謝辞).
