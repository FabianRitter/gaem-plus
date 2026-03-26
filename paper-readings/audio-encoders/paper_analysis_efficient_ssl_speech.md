# Deep Analysis: Efficient Training of Self-Supervised Speech Foundation Models on a Compute Budget

**Paper:** Liu, A.T., Lin, Y.-C., Wu, H., Winkler, S., & Lee, H.-y. (2025)  
**Venue:** ICASSP 2025 (IEEE International Conference on Acoustics, Speech and Signal Processing)  
**arXiv:** [2409.16295](https://arxiv.org/abs/2409.16295)

---

## 1. High-Level Summary (The "TL;DR")

- **Scaling laws for speech SSL:** The paper systematically investigates how model architecture shape, model size, data size, and SSL objective interact under a fixed compute (FLOPS) budget for self-supervised speech pre-training, finding that architecture shape and data diversity matter more than the choice of SSL objective (predictive vs. contrastive vs. generative).
- **Slim > Small:** Under matched compute and parameter budgets, "Slim" architectures (narrower hidden dimension, more Transformer layers) consistently outperform the "Small" architectures (wider, 3-layer) that have been the *de facto* standard for resource-constrained speech SSL research — a finding validated across HuBERT, wav2vec 2.0, and TERA.
- **Data size and the U-shaped curve:** Pre-training data volume remains critical even with data augmentation; iterating repeatedly over limited data degrades performance. The paper identifies U-shaped performance curves that reveal an optimal model size for any given compute budget, analogous to Chinchilla-type scaling observations in LLMs.

---

## 2. The Core Problem

### Context

Self-supervised learning (SSL) has become the dominant paradigm for learning general-purpose speech representations. Models like wav2vec 2.0, HuBERT, and WavLM, pre-trained on hundreds or thousands of hours of unlabeled speech, achieve strong downstream performance across ASR, speaker verification, emotion recognition, and other tasks when fine-tuned or probed with lightweight heads.

### The Gap

Despite this success, the speech SSL community has focused overwhelmingly on *designing new objectives* or *scaling up to larger models and data*, with little systematic investigation of *what actually matters* when you have a fixed compute budget. Specifically:

1. **Uncontrolled comparisons:** Existing benchmarks (e.g., SUPERB leaderboard) compare models trained with wildly different recipes — different data, different model sizes, different training durations — making it impossible to isolate the contribution of any single factor.
2. **Small model design is ad hoc:** Resource-constrained research typically defaults to a 3-layer Transformer ("Small"), a convention inherited from early work rather than derived from principled architectural analysis.
3. **Speech ≠ Text for scaling:** While the NLP/LLM community has extensive work on compute-optimal training (Kaplan et al., Hoffmann et al./Chinchilla), speech inputs are substantially longer than text tokens, and SSL objectives differ fundamentally from autoregressive language modeling. It is unclear whether LLM scaling insights transfer to speech.
4. **Data augmentation as a crutch:** Some SSL methods (e.g., wav2vec 2.0's masking with contrastive learning) implicitly augment data through random masking. It remains unclear whether this reduces the need for genuinely diverse pre-training data.

### Research Questions

The authors structure their investigation around four explicit questions:
- (Q1) Do self-supervised objectives matter under controlled conditions?
- (Q2) Does model architecture shape (depth vs. width) affect performance at fixed parameter count?
- (Q3) How does pre-training data size interact with data iteration?
- (Q4) Is there an optimal model size for a given compute budget?

---

## 3. Key Contributions

1. **Fair SSL objective comparison:** First fully controlled comparison of predictive (HuBERT), contrastive (wav2vec 2.0), and generative (TERA) SSL objectives under identical compute, parameter, and data constraints — revealing that objective choice influences but does not dominate performance.

2. **"Slim" architecture prescription:** Demonstration that deeper, narrower Transformer encoders (the "Slim" configuration) consistently outperform the shallow, wider "Small" configurations traditionally used in resource-constrained settings, validated across all three SSL objectives.

3. **Data size > data augmentation:** Evidence that pre-training data diversity (960 hours vs. 100 hours) remains essential even when SSL training implicitly augments data through masking/alteration, and that repeated iteration over limited data hurts.

4. **Compute-optimal model sizing for speech SSL:** Identification of U-shaped performance curves (training loss, validation loss, downstream ASR and ASV) as a function of model size at fixed FLOPS, suggesting an optimal model size of roughly 2× the Slim configuration for the tested budget — an early analogue of Chinchilla-style analysis for speech SSL.

5. **Practical training guidelines:** A set of actionable recommendations for researchers operating under compute constraints, backed by extensive ablations.

---

## 4. Methodology & Technical Details

### 4.1 Experimental Framework

The core design principle is **isolation of variables.** The authors fix all factors except the one under study:

- **SSL Objectives tested:**
  - **HuBERT** (predictive): Predicts cluster assignments (k-means on MFCC or intermediate representations) for masked frames. A two-iteration procedure: first iteration uses MFCC-based labels, second iteration uses labels derived from the first model's hidden states.
  - **wav2vec 2.0** (contrastive): Encodes speech with a CNN feature extractor, masks latent representations, and learns via a contrastive loss that distinguishes true quantized latents from distractors.
  - **TERA** (generative/reconstructive): Transformer Encoder Representations from Alteration — reconstructs original acoustic features from inputs corrupted by time alteration, frequency alteration, and magnitude alteration.

- **Pre-training data:** LibriSpeech (up to 960 hours), sampled at 16 kHz.
- **Evaluation:** SUPERB benchmark tasks including ASR (Automatic Speech Recognition), ASV (Automatic Speaker Verification), and other tasks, using frozen representations with lightweight prediction heads.

### 4.2 Architecture: "Small" vs. "Slim"

This is the paper's most impactful design choice. The authors contrast two model shapes:

| Configuration | Transformer Layers | Hidden Dim | Approx. Params |
|---|---|---|---|
| **Small** | 3 | Large (matching Base model width) | Budget-matched |
| **Slim** | ~12 (matching Base model depth) | Reduced | Budget-matched |

Both configurations are constrained to the same total parameter count and the same total training FLOPS. The "Small" configuration follows the convention in prior work (DistilHuBERT, MelHuBERT, etc.) of simply reducing the number of layers. The "Slim" configuration instead reduces the hidden dimension while preserving depth.

**Key insight:** The Slim model preserves the representational hierarchy of the full Base model (information flows through multiple Transformer layers) while sacrificing per-layer capacity. The Small model sacrifices depth (and thus the ability to build hierarchical representations) while keeping per-layer capacity high.

### 4.3 FLOPS Computation and Budget Control

The compute budget is operationalized as total training FLOPS (floating-point operations). For Transformer-based models, FLOPS scale approximately as:

**FLOPS ≈ 2 × N_params × N_tokens × N_steps**

(accounting for forward and backward passes)

Key constraint: When model size increases, the number of training steps decreases proportionally to maintain constant total FLOPS. This means larger models see fewer epochs of data — a critical trade-off.

### 4.4 Model Size Scaling Experiment

To investigate Q4 (optimal model size), the authors create a family of model sizes expressed as percentages of the Slim configuration (50%, 75%, 100%, 150%, 200%, 300%, 400%, 500%). All are pre-trained on 960 hours to maximize data diversity. Training steps are adjusted inversely with model size to keep FLOPS constant.

- The **smallest model (50% Slim)** trains for ~24.74 epochs.
- The **largest model (500% Slim)** trains for only ~3.25 epochs.

### 4.5 Data Size and Iteration Experiment

The authors systematically vary:
- Pre-training data: 1h, 10h, 100h, 460h, 960h of LibriSpeech
- Within fixed FLOPS, more data means fewer iterations per data point, and vice versa

A specific comparison contrasts "train on 100h with many iterations" vs. "train on 960h with fewer iterations" at matched FLOPS.

---

## 5. Experiments & Results

### 5.1 Datasets

- **Pre-training:** LibriSpeech (960 hours max), with subsets at 1h, 10h, 100h, and 460h.
- **Evaluation:** SUPERB benchmark — a suite of downstream tasks evaluated with frozen SSL representations and task-specialized lightweight prediction heads.

### 5.2 Baselines

The paper does not compare against external SOTA models. Instead, it treats each SSL method as both a system under study and a baseline for the others. The internal comparisons are:
- HuBERT Slim vs. HuBERT Small
- wav2vec 2.0 Slim vs. wav2vec 2.0 Small
- TERA Slim vs. TERA Small
- Cross-objective comparisons (HuBERT vs. wav2vec 2.0 vs. TERA)

### 5.3 Evaluation Metrics

From the SUPERB benchmark (Table 1 in the paper):
- **ASR:** Word Error Rate (WER) ↓
- **ASV:** Equal Error Rate (EER) ↓
- **Other SUPERB tasks:** Task-specific metrics (accuracy, F1, etc.) covering phoneme recognition, keyword spotting, intent classification, speaker identification, emotion recognition, etc.

### 5.4 Key Findings

#### SSL Objective Comparison (Q1)
Under fully controlled conditions (same architecture, same FLOPS, same data), HuBERT's predictive objective consistently outperforms wav2vec 2.0 and TERA across all SUPERB downstream tasks. However, wav2vec 2.0 and TERA show variable relative performance — each excels on different tasks. The authors conclude that while the SSL objective does matter, its impact is **less significant than model architecture or data choices**.

#### Architecture: Slim Outperforms Small (Q2)
Across all three SSL objectives, the Slim architecture (deeper, narrower) outperforms the Small architecture (shallower, wider) on essentially all downstream tasks. This is a robust finding: it holds for HuBERT, wav2vec 2.0, and TERA independently. The implication is strong — the standard 3-layer "Small" model convention in the literature leaves performance on the table. A narrower but deeper model is a better use of the same parameters and compute.

Critically, the **architecture choice has a larger impact on downstream performance than the choice of SSL objective.** A Slim TERA model can sometimes match or beat a Small HuBERT model, despite HuBERT being the stronger objective overall.

#### Data Size and Augmentation (Q3)
- Scaling pre-training data from 100h to 960h (nearly 10× increase) yields *limited* improvements in downstream performance under a fixed FLOPS budget. This counterintuitive finding likely reflects the FLOPS constraint: with more data, the model iterates fewer times over each example.
- However, the converse is more damaging: training on very small datasets (1h, 10h) with many iterations leads to substantial performance degradation, suggesting **overfitting to limited data** that SSL data augmentation (masking, alteration) cannot remedy.
- Data augmentation inherent to the SSL methods is **insufficient to compensate for lack of data diversity.** This is a key negative result.

#### Compute-Optimal Model Size (Q4)
The U-shaped performance curves (Figure 2 in the paper) reveal a clear trade-off:
- **Too small a model:** Underfits despite many training epochs.
- **Too large a model:** Underfits because it sees too few epochs of data within the FLOPS budget.
- **Sweet spot:** For all three SSL methods, the optimal model size is approximately **200% of the Slim configuration** (i.e., roughly twice the baseline Slim model).

This U-shape is observed consistently across training loss, validation loss, ASR downstream performance (WER), and ASV downstream performance (EER), providing strong evidence that this is a genuine phenomenon rather than a metric-specific artifact.

---

## 6. Strengths

- **Rigorous experimental controls:** The paper's strongest contribution is methodological. By isolating variables (objective, architecture, data, compute) one at a time while holding others constant, it provides the kind of apples-to-apples comparison that the speech SSL community has lacked. This is genuinely valuable.

- **Actionable and practical:** The Slim-over-Small finding is immediately actionable. Any researcher working with small speech SSL models can re-allocate their parameter budget to a deeper, narrower architecture with high confidence of improvement.

- **Cross-objective validation:** Every claim is validated across all three SSL methods (HuBERT, wav2vec 2.0, TERA), strengthening the generalizability of the findings.

- **Addresses a real gap:** The NLP community has extensive scaling law literature (Kaplan et al., Chinchilla, etc.), but the speech community has had almost nothing comparable. This paper begins to fill that gap.

- **Honest about scope:** The authors are careful to state that their goal is *not* to beat SUPERB SOTA but to understand training dynamics. This framing avoids misleading readers about the paper's contribution.

- **The U-shaped curve is a compelling finding:** It provides a speech-specific analogue to compute-optimal scaling, and the consistency across multiple metrics and methods adds credibility.

---

## 7. Weaknesses & Limitations

- **Limited FLOPS budget tested:** The paper operates at a single (relatively small) compute budget. It is unclear whether the findings (especially the 200% optimal size) generalize to larger budgets. Chinchilla scaling for LLMs showed that optimal ratios shift with total compute — the same likely applies here. A single budget point is insufficient to derive actual scaling laws.

- **Only LibriSpeech:** All experiments use LibriSpeech, a clean read-speech English corpus from audiobooks. The findings may not transfer to noisy/spontaneous speech, multilingual settings, or non-speech audio (music, environmental sounds). The data diversity dimension is explored only within LibriSpeech subsets, not across genuinely different domains.

- **Slim architecture is underspecified:** The paper does not provide a precise methodology for choosing the Slim configuration's layer count vs. hidden dimension trade-off. "More layers, narrower" is a useful heuristic, but the exact design point appears ad hoc (matching the Base model's depth). A sweep over multiple depth-width trade-offs at fixed parameter count would strengthen this claim.

- **No modern SSL methods:** The comparison covers HuBERT, wav2vec 2.0, and TERA — all circa 2020–2021. More recent methods like WavLM, data2vec, data2vec 2.0, BestRQ, and SPIN are absent. Whether the findings (especially objective-agnosticism) extend to newer methods is unknown.

- **SUPERB with frozen features only:** The evaluation protocol freezes the SSL encoder and trains lightweight heads. This is one valid evaluation paradigm, but fine-tuning the entire model (the other common paradigm) may yield different conclusions about optimal architecture shape. Deeper models might benefit more from fine-tuning than wider models, potentially amplifying or attenuating the Slim advantage.

- **No wall-clock analysis:** FLOPS is a useful proxy for compute, but real-world efficiency depends on hardware utilization, memory bandwidth, and parallelization. A 12-layer Slim model and a 3-layer Small model may have very different GPU utilization profiles. The paper does not report actual training time or memory consumption.

- **HuBERT's clustering overhead is not accounted for:** HuBERT requires an offline k-means clustering step (and optionally a second iteration). This compute cost is not included in the FLOPS budget, giving HuBERT an implicit advantage in the comparisons.

- **Limited downstream diversity:** While SUPERB covers multiple tasks, the detailed analysis focuses primarily on ASR and ASV. The paper could have explored whether the Slim advantage is task-dependent (e.g., stronger for content tasks vs. speaker tasks).

- **No Conformer or non-Transformer architectures:** All models use standard Transformer encoders. Conformer-based SSL (which adds convolution modules for local feature capture) might alter the depth-vs-width trade-off.

---

## 8. Practical Implications & Applications

### For Resource-Constrained Research Labs
The most direct beneficiary is any researcher or organization training speech SSL models under compute constraints (limited GPU hours, academic budgets, edge deployment targets). The prescription is clear: favor deeper, narrower architectures over the conventional shallow-wide design.

### For Speech SSL Practitioners
- When choosing between SSL objectives under a budget, the architectural choices matter more. Invest engineering effort in architecture search rather than objective engineering.
- Do not rely on data augmentation inherent to the SSL method as a substitute for genuine data diversity. If possible, curate a diverse pre-training corpus even if it means smaller total hours.
- If pre-training on limited data is unavoidable, be aware that excessive iteration will degrade the representations.

### For Efficient Model Design
The U-shaped curve finding has practical implications for model deployment: if you have a fixed compute budget for pre-training, there exists a sweet spot model size that maximizes downstream utility. Over-parameterizing beyond this point wastes compute on a model that cannot be adequately trained within the budget.

### For Benchmarking
The paper highlights a critical problem with existing benchmarks: uncontrolled training recipes make it impossible to attribute performance differences to any single factor. Future benchmarks and competitions should mandate controlled settings or at minimum report FLOPS alongside performance.

---

## 9. Future Work & Open Questions

### Directions Suggested by the Paper's Own Logic

1. **Multi-budget scaling laws for speech SSL:** The most obvious extension is repeating the U-shaped curve analysis at multiple FLOPS budgets (e.g., 1×, 10×, 100× the current budget) to derive actual power-law scaling relationships analogous to Chinchilla. This would allow predicting optimal model size for any given budget.

2. **Architecture search beyond depth/width:** The Slim-vs-Small comparison is a 2-point comparison on the depth-width Pareto frontier. A finer-grained sweep (e.g., 6 layers at medium width, 24 layers at very narrow width, etc.) would better characterize the optimal shape. Including Conformer variants would be essential.

3. **Modern SSL methods:** Extending the controlled comparison to WavLM, data2vec 2.0, BestRQ, and other recent methods would test whether the "objective doesn't matter much" conclusion holds broadly.

4. **Multilingual and multi-domain data:** Testing whether the data size findings hold when pre-training data spans multiple languages and acoustic conditions.

5. **Fine-tuning evaluation:** Repeating the experiments with full model fine-tuning (not just frozen representations) to check whether the Slim advantage persists.

### Implications for Model Merging in Speech/Audio Foundation Models

This paper has several direct implications for your research program on task-specific projector merging for speech-to-LLM adaptation (SpeechMapper direction):

1. **Architecture of the speech encoder feeding the projector matters:** If you are using a standard SSL model (like those from SeamlessM4T-v2 or wav2vec 2.0/HuBERT directly), the paper suggests that the layer-wise representation quality is heavily influenced by the depth-vs-width trade-off. For model merging, this means: (a) features extracted from different layers of deeper models may have more hierarchical structure, which could affect how well task-specific projectors diverge after adaptation; and (b) if you train smaller SSL encoders as part of any ablation, use Slim configurations.

2. **SSL objective agnosticism is encouraging for merging:** If the SSL objective is less important than architecture/data, this suggests that the projector (which operates *downstream* of the SSL encoder) is the right place to invest merging effort. The SSL encoder's internal representations are relatively robust to objective choice, so merging projectors rather than encoders is a sound strategy.

3. **Data diversity for task-specific adaptation:** The paper's finding that data diversity matters more than iteration count has implications for Stage 2 adaptation. When training task-specific projectors (ASR, speech translation, spoken QA), using diverse data per task is more important than running many epochs on limited data. This may also help ensure sufficient divergence between task-specific projectors — a key concern for your merging experiments.

4. **Compute-optimal projector sizing:** The U-shaped curve concept could be applied to the projector itself. There may be an optimal projector size (number of Transformer layers, hidden dimension) for a given adaptation compute budget. Over-parameterizing the projector might lead to underfitting if the adaptation budget is small.

5. **Permutation alignment within Slim vs. Small architectures:** The paper's finding that depth matters more than width has implications for permutation alignment methods applied to Transformer blocks within projectors. Deeper architectures create more alignment opportunities (more layers to align), but each layer has a smaller feature space, which may make the alignment problem better-conditioned. This is a concrete research question worth investigating.

6. **Applicability to audio/music:** The paper is limited to speech (LibriSpeech). For any extension to music or environmental audio SSL, the data diversity finding is likely even more critical, since audio domains are far more heterogeneous than read English speech. The Slim-vs-Small finding should be validated for audio SSL encoders (e.g., Audio-MAE, BEATs, ATST).

---

## 10. What I Would Have Done Differently & Extensions for a New Publication

### Critique of Experimental Design

1. **I would have tested at least 3 FLOPS budgets (e.g., 1×, 5×, 25×)** to move beyond a single-point observation toward actual scaling laws. A single budget point identifies a phenomenon but cannot characterize it mathematically. Even 3 budget points would allow fitting a power law and predicting optimal configurations for unseen budgets.

2. **I would have included WavLM and data2vec 2.0** as SSL objectives. Omitting the most competitive methods weakens the generality claims. WavLM's denoising objective, in particular, represents a genuinely different inductive bias from the three methods tested.

3. **I would have conducted a proper depth-width Pareto sweep.** Testing only 2 architectural configurations (Small = 3 layers, Slim = ~12 layers) is insufficient. I would sweep: {3, 6, 9, 12, 18, 24} layers at matched parameter counts, plotting a full Pareto frontier of depth vs. downstream performance. This would reveal whether the Slim advantage is monotonically increasing with depth or whether there is a sweet spot.

4. **I would have reported wall-clock time and memory.** FLOPS is a useful theoretical measure, but practitioners care about GPU-hours and peak memory. A 12-layer model with 256-dim hidden states might have worse GPU utilization than a 3-layer model with 768-dim hidden states due to sequential layer computation. This practical consideration is absent.

5. **I would have included at least one non-English and one noisy-speech evaluation** to test generalization beyond LibriSpeech's clean, read-speech English domain.

### Extension for a New Publication

**Title:** *Scaling Laws for Speech Self-Supervised Learning: Compute-Optimal Model Design Across Objectives, Architectures, and Data Domains*

**Core extension:**
- Repeat the paper's controlled framework at 5+ FLOPS budgets spanning 2 orders of magnitude.
- Fit parametric scaling laws (loss as a function of N_params and D_data at fixed compute, following the Chinchilla formulation adapted for speech).
- Extend to Conformer architectures and modern SSL methods (WavLM, BestRQ, data2vec 2.0).
- Evaluate on multilingual data (VoxPopuli, CommonVoice) and noisy/spontaneous speech (CHiME, Fisher).
- Include fine-tuning evaluation alongside frozen-representation probing.
- Derive concrete "recipes" — lookup tables mapping budget → optimal (depth, width, data_hours, objective) — that the community can use directly.

**For my own research in model merging for speech/audio foundation models:**
- Use the scaling law framework to identify compute-optimal projector configurations before running expensive merging experiments.
- Test whether the Slim-vs-Small finding affects merging success (hypothesis: deeper projectors may be harder to merge naively due to more layers requiring alignment, but permutation alignment methods should handle this well).
- Apply the data diversity insight to ensure task-specific projectors are trained on sufficiently diverse data to guarantee divergence, making the merging experiments scientifically meaningful.
- Investigate whether the "objective agnosticism" finding extends to the projector's adaptation loss — if so, simpler MSE-based projector training (as in SpeechMapper Stage 1) is on solid ground relative to more complex contrastive or adversarial alternatives.

---

## Summary Table

| Aspect | Assessment |
|---|---|
| **Novelty** | Moderate — adapts NLP scaling law methodology to speech SSL |
| **Rigor** | High — careful variable isolation, cross-method validation |
| **Impact** | High for practical speech SSL research under compute constraints |
| **Presentation** | Clear and well-structured |
| **Completeness** | Moderate — limited to one budget, one dataset, three methods |
| **Reproducibility** | Good — uses public data (LibriSpeech) and public benchmark (SUPERB) |
| **Venue fit** | Appropriate for ICASSP (4-page format limits depth of analysis) |

**Overall assessment:** A solid, methodologically careful empirical study that provides actionable insights for the speech SSL community. Its main limitation is scope — a single compute budget and limited method/data diversity prevent it from achieving the generality of a true "scaling laws" paper. Nevertheless, the Slim-over-Small finding and the U-shaped curve observation are valuable contributions that should influence standard practice.
