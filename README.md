# WARP: Guaranteed Inner-Layer Repair of NLP Transformers

[![arXiv](https://img.shields.io/badge/arXiv-2604.00938-b31b1b.svg)](https://arxiv.org/abs/2604.00938)

Code for **[WARP: Guaranteed Inner-Layer Repair of NLP Transformers](https://arxiv.org/pdf/2604.00938)**.

**Authors:** Hsin-Ling Hsu, Min-Yu Chen, Nai-Chia Chen, Yan-Ru Chen, Yi-Ling Chang, Fang Yu

---

Transformer-based NLP models remain vulnerable to adversarial perturbations, yet existing repair methods face a fundamental trade-off: gradient-based approaches offer flexibility but lack verifiability and often overfit; methods that do provide repair guarantees are restricted to the final layer or small networks, significantly limiting the parameter search space available for repair. We present WARP (Weight-Adjusted Repair with Provability), a constraint-based repair framework that extends repair beyond the last layer of Transformer models. WARP formulates repair as a convex quadratic program derived from a first-order linearization of the logit gap, enabling tractable optimization over a high-dimensional parameter space. Under the condition that the first-order approximation holds, this formulation induces three per-sample guarantees: (i) a positive margin constraint ensuring correct classification on repaired inputs, (ii) preservation constraints over a designated remain set, and (iii) a certified robustness radius derived from Lipschitz continuity. To ensure feasibility across varying model architectures, we introduce a sensitivity-based preprocessing step that conditions the optimization landscape accordingly. We further show that the iterative optimization procedure converges to solutions satisfying all repair constraints under mild assumptions. Empirical evaluation on encoder-only Transformers with varying layer architectures validates that these guarantees hold in practice while improving robustness to adversarial inputs. Our results demonstrate that guaranteed, generalizable Transformer repair is achievable through principled constraint-based optimization.

---

- **Encoder-only** (DistilBERT / BERT, SST-2 / RTE): WARP + LoRA / Full FT baselines
- **Decoder-only** (GPT-2 / GPT-2-medium / Qwen2.5-0.5B, SST-2 / RTE): WARP + LoRA / Full FT baselines
- **PRoViT** comparison (six decoder-only settings, Supplementary G)

## Structure

```
code/
├── README.md
├── data/
│   ├── dev_ann.json          # AdvGLUE repair set
│   └── test_ann.json         # AdvGLUE attack-generalization set
├── encoder/
│   ├── encoder_sst2_experiments.ipynb
│   └── encoder_rte_experiments.ipynb
├── decoder/
│   ├── decoder_experiments.ipynb   # WARP + LoRA/Full FT, GPT-2/GPT-2-medium/Qwen2.5-0.5B × SST-2/RTE
│   ├── provit_baseline.ipynb       # PRoViT comparison
│   └── results_bundled/            # finished result_*.pkl
```

## Environment

- **Platform**: Google Colab recommended.
- **Packages**: each notebook's first cell `pip install`s everything it
  needs.
- **GPU**: Experiments were run on an A100-SXM4-80GB.
- **Env vars**: `GUROBI_WLS_LICENSE` is only needed for the LP / FT+LP
  variants of `provit_baseline.ipynb` (free academic license at
  gurobi.com/academia).

## Data

`data/dev_ann.json` and `data/test_ann.json` are the annotated dev and test
sets of [AdvGLUE](https://adversarialglue.github.io/) (Wang et al., 2021),
redistributed unmodified under the
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.
The original (benign) sentences come from the GLUE benchmark (SST-2, RTE);
please also respect the terms of the original GLUE datasets.

If you use these files, please cite AdvGLUE:

```bibtex
@inproceedings{wang2021adversarial,
  title={Adversarial GLUE: A Multi-Task Benchmark for Robustness Evaluation of Language Models},
  author={Wang, Boxin and Xu, Chejian and Wang, Shuohang and Gan, Zhe and Cheng, Yu and Gao, Jianfeng and Awadallah, Ahmed Hassan and Li, Bo},
  booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
  year={2021}
}
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{hsu2026warp,
  title={WARP: Guaranteed Inner-Layer Repair of NLP Transformers},
  author={Hsu, Hsin-Ling and Chen, Min-Yu and Chen, Nai-Chia and Chen, Yan-Ru and Chang, Yi-Ling and Yu, Fang},
  journal={arXiv preprint arXiv:2604.00938},
  year={2026}
}
```
