# Critical Analysis: MERT -- Acoustic Music Understanding Model with Large-Scale Self-Supervised Training

**Paper:** Li, Y., Yuan, R., Zhang, G., Ma, Y., et al. (2024). *MERT: Acoustic Music Understanding Model with Large-Scale Self-Supervised Training.* Published at ICLR 2024.  
**Venue:** International Conference on Learning Representations (ICLR) 2024  
**ArXiv:** [2306.00107](https://arxiv.org/abs/2306.00107)

---

## 1. High-Level Summary (The "TL;DR")

- MERT adapts the HuBERT-style masked language modelling (MLM) paradigm from speech SSL to music audio by introducing a dual-teacher framework: an **acoustic teacher** (RVQ-VAE / EnCodec or K-means) that provides discretised acoustic pseudo-labels, and a **musical teacher** (Constant-Q Transform) that supplies pitch and harmonic reconstruction targets. The combination is trained via a multi-task loss.
- The authors scale this framework from 95M to 330M parameters, documenting and resolving significant training instability issues (gradient explosions under fp16), and evaluate on 14 diverse MIR downstream tasks under a frozen-backbone probing protocol. MERT-330M matches or exceeds the aggregate performance of 10 prior specialised SOTA models while being roughly 15x smaller than Jukebox-5B.
- The paper provides a systematic ablation of teacher model choices and loss weighting, yielding practical guidelines for future music SSL pre-training, and releases code, model weights, and a reproducible evaluation benchmark.

---

## 2. The Core Problem

Music Information Retrieval (MIR) encompasses a wide array of tasks: tagging, key detection, genre classification, beat tracking, pitch estimation, source separation, emotion recognition, and more. Historically, each task has been addressed with specialised models and handcrafted features. The fundamental problem MERT addresses is:

**There exists no general-purpose, open-source, computationally affordable pre-trained model for acoustic music understanding.**

Several factors contribute to this gap:

1. **Data scarcity and copyright constraints.** Large-scale annotated music datasets are expensive to build and legally restricted, making supervised pre-training impractical at scale.
2. **Failure of speech SSL to transfer directly to music.** Speech SSL models like HuBERT rely on MFCC-based clustering to generate pseudo-labels. MFCCs capture formant and timbre information suitable for single-pitch speech signals but are poor at encoding the polyphonic, harmonic, and tonal structures fundamental to music.
3. **Existing music foundation models are either too large or not open.** Jukebox-5B provides strong representations but at enormous computational cost (weeks of inference on a single GPU for a moderately-sized dataset). Other models (MULE, MusiCNN, CLMR) either rely on supervised labels, cover only tagging tasks, or lack public checkpoints.
4. **Training instability at scale.** Scaling encoder-only audio Transformers beyond 100M parameters introduces gradient instability under mixed-precision training that is more severe than in text or vision domains, and for which standard remedies (DeepNorm) prove ineffective.

---

## 3. Key Contributions

1. **A multi-task predictive SSL paradigm for music.** MERT introduces a dual-teacher design combining an acoustic pseudo-label loss (cross-entropy on discrete codes) with a musical reconstruction loss (MSE on CQT spectrograms). This is the central methodological contribution.
2. **Systematic ablation of teacher model design.** The paper evaluates K-means on various feature combinations (MFCC, log-Mel + Chroma, MFCC + CQT) and RVQ-VAE (EnCodec) as acoustic teachers, and CQT as the musical teacher, providing a clear empirical decision route for future practitioners.
3. **Strategies for stable large-scale acoustic model training.** The authors document that Pre-LN combined with attention relaxation (from WavLM) resolves training crashes that neither gradient clipping reduction nor DeepNorm could fix, enabling scaling to 330M parameters.
4. **Comprehensive open-source evaluation on 14 MIR tasks.** The evaluation goes well beyond the typical music tagging benchmark, covering frame-level tasks (pitch, beat, emotion, key, genre, instrument, vocal technique, singer ID) and sequence-level tasks (source separation), under a strict probing protocol.
5. **An open, reproducible, and lightweight model release.** MERT-95M and MERT-330M are publicly available, providing the community with a practical alternative to billion-parameter models.

---

## 4. Methodology and Technical Details

### 4.1 Architecture Overview

MERT inherits the HuBERT architecture:

- **1D Convolutional Feature Extractor.** A multi-layer 1D CNN that converts raw audio waveforms into a sequence of local feature vectors, analogous to the convolutional front-end in wav2vec 2.0 / HuBERT.
- **Transformer Encoder.** A stack of self-attention layers with convolutional relative positional embeddings (following wav2vec 2.0). The 95M model uses 12 Transformer layers; the 330M model uses 24 layers.
- **Projection Heads.** Separate linear projection heads on top of the Transformer output for the acoustic and musical prediction tasks.

The input audio is randomly truncated to 5-second segments. A span masking strategy (following HuBERT conventions) masks contiguous spans of the feature sequence. The model must predict the pseudo-labels at masked positions only.

### 4.2 The Dual-Teacher Framework

The key innovation is the design of two complementary teachers that supply pseudo-labels covering different aspects of musical information:

#### Acoustic Teacher

Two options are explored:

**Option A: K-means on handcrafted features.**
K-means clustering is applied to log-Mel spectral features (300 centroids, 229-dimensional) and Chroma features (200 centroids, 264-dimensional). The product of the two codebooks yields 60,000 classes. This captures timbre (via log-Mel) and harmonic content (via Chroma) but is expensive to scale.

**Option B: RVQ-VAE (EnCodec).**
EnCodec is an 8-layer Residual Vector Quantised VAE that converts 24 kHz audio into a matrix of discrete codes:

$$z_{\text{enc}} \in [C]^{L \times 8}$$

where $L$ is the number of frames (75 Hz output rate, so $L = 375$ for 5 seconds), and each of the 8 codebook layers has $C = 1024$ entries. The row $z_{\text{enc}}[t, :]$ gives 8 hierarchical discrete codes for frame $t$. EnCodec is attractive because it is a pre-trained neural codec, scales trivially (no K-means fitting), and its discrete codes demonstrably capture enough acoustic information for high-fidelity waveform reconstruction.

#### Musical Teacher (CQT)

The Constant-Q Transform is a frequency-domain representation where bin widths are proportional to frequency rather than fixed (as in the STFT). This gives each octave the same number of frequency bins, directly encoding the logarithmic pitch structure of Western music. The CQT spectrogram serves as a continuous reconstruction target (not a discrete pseudo-label), providing an explicit harmonic and pitch inductive bias.

### 4.3 Training Objectives

The total loss is a linear combination:

$$\mathcal{L} = \alpha \cdot \mathcal{L}_H + \mathcal{L}_{\text{CQT}}$$

where:

**Acoustic MLM Loss** $\mathcal{L}_H$: This is the standard HuBERT-style Noise Contrastive Estimation (NCE) loss over discrete pseudo-labels. For each masked position $t$, the model output $o_t$ is linearly projected via $T(\cdot)$ and compared against codebook embeddings $e_c$ via cosine similarity:

$$p_f(c \mid x', t) = \frac{\exp(\text{sim}(T(o_t), e_c) / \tau)}{\sum_{c'=1}^{C} \exp(\text{sim}(T(o_t), e_{c'}) / \tau)}$$

with temperature $\tau = 0.1$. When using EnCodec, this loss is computed over all 8 codebook layers simultaneously, meaning the model predicts 8 separate discrete targets per frame.

**Musical Reconstruction Loss** $\mathcal{L}_{\text{CQT}}$: A mean squared error between the predicted CQT spectrogram and the ground-truth CQT at masked positions:

$$\mathcal{L}_{\text{CQT}} = \sum_{t \in M} \| z_{\text{cqt},t} - f_{\text{cqt}}(x')_t \|^2$$

The weight $\alpha$ is searched over $\{1, 2, 5\}$, with $\alpha = 1$ found optimal.

### 4.4 In-Batch Noise Mixup

An augmentation strategy randomly samples short audio segments from within the same mini-batch and adds them at random positions within training clips with a probability of 0.5. This is conceptually similar to mixup but applied as additive noise, forcing the model to learn representations that are robust to interfering sources. This is motivated by the polyphonic "cocktail party" nature of music.

### 4.5 Training Stability

Scaling to 330M with fp16 precision introduces instability:

- **Pre-LN** (placing LayerNorm before the attention/FFN block rather than after) is a necessary but insufficient condition.
- **DeepNorm** (modified residual scaling and initialisation for deep Transformers, from Wang et al. 2022) causes model collapse around 20K steps.
- **Attention relaxation** (an additional learned scale constant in the softmax denominator, from WavLM, Chen et al. 2021b) resolves the overflow problem and enables stable training beyond 100K steps.

This is a practically valuable finding, as training instability in audio Transformers is under-documented compared to text/vision.

---

## 5. Experiments and Results

### 5.1 Datasets

14 downstream tasks spanning:

| Task | Dataset(s) |
|------|-----------|
| Music tagging | MagnaTagATune (MTT), MTG-Jamendo |
| Key detection | Giantsteps, Giantsteps-MTG-keys |
| Genre classification | GTZAN, MTG-Genre |
| Emotion regression | Emomusic |
| Instrument classification | Nsynth, MTG-instrument |
| Pitch classification | Nsynth |
| Vocal technique detection | VocalSet |
| Singer identification | VocalSet |
| Beat tracking | GTZAN Rhythm |
| Source separation | MUSDB18 |

### 5.2 Baselines

- **Supervised:** MusiCNN (supervised tagging pre-training on Million Song Dataset)
- **Contrastive:** CLMR, MULE
- **Generative:** Jukebox-5B / JukeMIR
- **Speech SSL re-trained on music:** HuBERT$_{\text{music}}$, data2vec$_{\text{music}}$

### 5.3 Evaluation Protocol

Frozen backbone probing only (no fine-tuning). A single-hidden-layer MLP (512 units) is trained on the extracted representations for frame-level tasks; a 3-layer BiLSTM is used for source separation. Hyperparameter search is limited to learning rate $\in \{1\text{e-4}, 5\text{e-4}, 1\text{e-3}, 5\text{e-3}, 1\text{e-2}\}$.

### 5.4 Key Quantitative Results

- **MERT-330M achieves an average score of 64.7** across all 14 tasks, matching the aggregated best results from 10 different specialised models (which collectively score 64.5).
- MERT-330M is the new SOTA on 4 individual metrics while using only 6.6% the parameters of Jukebox-5B.
- MERT-95M variants already achieve average scores of 62.9--63.7 at only 1.9% of Jukebox's parameter count.
- Key detection accuracy on Giantsteps improves from 15.1% (HuBERT with MFCC teacher) to 65.0% (MERT-95M with K-means + CQT) to 65.6% (MERT-330M), demonstrating the impact of incorporating pitch-aware teachers.
- Source separation SDR (vocals) reaches 5.3--5.6 for MERT variants vs. 5.1 for Jukebox-5B, though far below the specialised SOTA of 9.3 (Hybrid Transformers).

### 5.5 Ablation Studies

The ablation studies are among the most valuable parts of this paper:

1. **Teacher selection:** MFCC-only K-means is catastrophically poor at key detection (15.1% accuracy). Adding Chroma features jumps accuracy to 55.1%. Adding CQT as a separate musical teacher raises it further to 65.0%.
2. **RVQ-VAE codebook analysis:** Using only a single codebook layer (top or bottom) degrades performance substantially. Using all 8 codebook layers is optimal. Randomly sampling one codebook per batch is a viable memory-saving alternative with minor quality loss.
3. **Musical loss weight:** $\alpha = 1$ is optimal. Increasing to $\alpha = 2$ or $\alpha = 5$ degrades key detection and emotion recognition, suggesting the acoustic loss should not be dominated by the musical reconstruction objective.
4. **In-batch noise mixup:** Probability 0.5 helps the RVQ-VAE teacher setting (raising average from 66.9 to 68.8 on the ablation subset) but slightly hurts the K-means setting, suggesting an interaction between augmentation and teacher type.
5. **Scaling:** 330M generally improves over 95M, but with an inverse-scaling effect on beat tracking ($F_1$ drops from 88.3 to 87.9), indicating incomplete training stabilisation.

---

## 6. Strengths

- **Principled and well-motivated dual-teacher design.** The separation of acoustic information (timbre, spectral envelope) from musical information (pitch, harmony) is conceptually clean, empirically validated, and grounded in the known shortcomings of speech SSL features for music.
- **Thorough ablation and transparency.** The paper systematically varies every major design choice (feature type, codebook configuration, loss weight, augmentation probability, normalisation scheme) and reports the full results. The decision route in Sections 5.2 and 5.3 is a model for how empirical SSL papers should be written.
- **Breadth of evaluation.** 14 tasks covering tagging, classification, regression, sequence labelling, and source separation. The probing-only protocol is the correct evaluation for representation quality (as opposed to conflating representation quality with fine-tuning capacity).
- **Practical training stability analysis.** The training instability documentation (Appendix B.3) with gradient norm and loss scale curves is candid and genuinely useful for practitioners scaling audio Transformers.
- **Open and reproducible.** Code, weights, and a standardised evaluation benchmark (MARBLE) are released. The MERT-95M-public variant trained entirely on open data addresses reproducibility.
- **Computational efficiency.** Achieving competitive performance at 95M--330M parameters compared to Jukebox-5B is a meaningful practical contribution.

---

## 7. Weaknesses and Limitations

- **5-second context window is a significant constraint.** Music has hierarchical structure at multiple timescales (phrases, sections, entire songs). A 2-bar context cannot capture verse-chorus structure, long-range key modulations, or form-level patterns. While the authors acknowledge this, the evaluation also does not stress-test long-range understanding: tasks like key detection and genre classification are applied to full tracks but by averaging short-window embeddings, which may mask the limitation.

- **The CQT teacher encodes a strong Western music bias.** The Constant-Q Transform with equal temperament bin spacing assumes 12-tone equal temperament tuning. For music traditions with microtonal intervals (Arabic maqam, Turkish makam, Indian raga, Javanese gamelan), this teacher injects a misaligned inductive bias. The paper does not discuss or evaluate cross-cultural generalisation.

- **The probing protocol, while principled, undersells representation quality.** The gap between probing and fine-tuning performance is well-documented in speech SSL. Reporting only probing results makes it difficult to compare with methods that report fine-tuning numbers, and may underestimate MERT's true capacity.

- **Source separation performance is far below specialised SOTA.** MERT achieves 5.3--5.6 SDR (vocals) vs. 9.3 for the specialised SOTA (Hybrid Transformers). This is a large gap that suggests the frozen SSL features are insufficient for this task, and the inclusion of source separation in the "14-task evaluation" somewhat inflates the breadth claim.

- **Training data is not fully characterised.** The 160K-hour dataset is described only as "mined from the Internet." The genre, geographic, and temporal distribution are unknown. The MERT-95M-public variant is trained on Music4All (910 hours of mainly pop music), which is acknowledged to lack diversity, but the private dataset receives no similar characterisation.

- **No iterative teacher refinement.** HuBERT uses a two-stage training where the learned representations from the first stage are used to train a better K-means teacher for the second stage. The paper finds this does not help for MFCC features but does not explore iterative refinement with the EnCodec teacher or with learned features from MERT itself. This leaves a potentially impactful design axis unexplored.

- **Reliance on an external pre-trained codec (EnCodec) as the acoustic teacher.** The quality and biases of EnCodec are inherited silently. EnCodec was trained primarily on speech and general audio; its adequacy as a music codec (particularly for complex polyphonic textures) is assumed rather than validated. Furthermore, the EnCodec teacher is frozen; co-training or fine-tuning it on music data might yield better pseudo-labels.

- **The "average score" metric across heterogeneous tasks is ad hoc.** Averaging ROC-AUC, accuracy, $R^2$, F1, and SDR into a single number treats fundamentally different metrics as commensurable. A model that is 2% better on tagging ROC and 4 dB worse on source separation SDR could appear "average." The paper would benefit from a more principled aggregation or at least a ranking-based analysis.

- **Inverse scaling on beat tracking is not resolved.** The 330M model is slightly worse than 95M on beat tracking, suggesting that training instability is not fully addressed and that larger models may memorise different distributional artifacts.

---

## 8. Practical Implications and Applications

- **Unified MIR feature extractor.** Studios, streaming platforms (Spotify, Apple Music), and research labs can use a single MERT model to extract features for tagging, genre classification, mood estimation, and instrument detection, replacing a zoo of task-specific models and handcrafted feature pipelines.

- **Lightweight deployment.** At 95M parameters, MERT can run inference on consumer GPUs in real time, enabling integration into music production DAWs, real-time recommendation engines, and mobile music apps.

- **Music education and analysis tools.** The pitch-aware representations could power automated feedback systems for music students (intonation analysis, chord recognition) or musicological analysis tools.

- **Content moderation and rights management.** Robust music representations could assist in content identification, cover detection, and rights management for streaming platforms.

- **Downstream model bootstrapping.** For tasks like source separation or beat tracking where MERT's probing performance is limited, the representations can serve as initialisation for fine-tuned systems, reducing labelled data requirements.

- **Redistribution of musical knowledge.** As the authors note, releasing a pre-trained model allows sharing of learned musical representations without distributing copyrighted audio data, which is particularly important given the legal landscape around music datasets.

---

## 9. Future Work and Open Questions

### General Research Directions

1. **Longer context training.** Extending the training context from 5 seconds to 30+ seconds (or using hierarchical approaches) would enable MERT to capture section-level and form-level musical structure. This is perhaps the most obvious improvement.

2. **Cross-cultural teacher design.** Replacing or augmenting the CQT teacher with a tuning-agnostic pitch representation (e.g., a learned filterbank or a CQT with non-equal-temperament bin spacing) would address the Western music bias.

3. **Iterative teacher refinement with learned features.** Using MERT's own intermediate representations to train improved K-means or VQ teachers for a second round of pre-training (as in HuBERT's iterative scheme) could improve representation quality, particularly if the acoustic teacher were retrained on MERT's learned feature space.

4. **Fine-tuning evaluation.** Full fine-tuning and parameter-efficient fine-tuning (LoRA, adapters) evaluations would better characterise the model's ceiling performance and utility for downstream practitioners.

5. **Scaling laws.** The paper trains only 95M and 330M. Charting scaling laws for music SSL (how does downstream performance scale with model size and data size?) would inform resource allocation decisions.

6. **Multimodal extensions.** Integrating text supervision (e.g., music captions, lyrics) alongside the acoustic and musical teachers could yield representations useful for cross-modal retrieval.

### Implications for Model Merging Research on Speech/Audio Foundation Models

This is where MERT connects directly to your research programme, Fabian. Several observations:

1. **MERT's dual-teacher design creates natural task decompositions.** The acoustic teacher (EnCodec) and musical teacher (CQT) encourage different subsets of the Transformer's capacity to specialise. This is analogous to training separate task-specific projectors from a shared backbone. One could investigate whether the internal representations learned under each teacher objective occupy different subspaces, and whether these subspaces can be independently manipulated via model merging.

2. **Multiple MERT variants as merging candidates.** The paper provides models trained with different teachers (K-means, RVQ-VAE, with/without CQT). These are structurally identical models trained from different initialisation and supervision signals, making them natural candidates for weight-space merging experiments. Key question: do MERT-95M$^{\text{K-means}}$ and MERT-95M$^{\text{RVQ-VAE}}$ lie in the same loss basin? Can you merge them to get a model that inherits the strengths of both teacher types?

3. **Permutation alignment within MERT.** MERT's Transformer encoder layers exhibit the same permutation symmetries you study. Applying permutation alignment before merging MERT variants trained with different teachers could test whether your methods work for music SSL, extending your results beyond speech.

4. **The dual-loss structure as a case study for multi-objective merging.** Merging models that were trained with different relative weights of the acoustic vs. musical loss ($\alpha = 1$ vs. $\alpha = 2$ vs. $\alpha = 5$) would test whether weight-space interpolation can smoothly traverse the Pareto frontier of the multi-objective loss landscape.

5. **MERT as a speech encoder alternative in SpeechMapper.** In your SpeechMapper projector merging project, MERT could serve as an alternative encoder to SeamlessM4T-v2-large for music-specific tasks. If you train SpeechMapper projectors with both SeamlessM4T and MERT as encoders, you could study whether merged projectors can handle both speech and music inputs, a direction toward a genuinely universal audio-to-LLM interface.

6. **Probing the linear mode connectivity of music SSL models.** MERT provides a controlled experimental setup: same architecture, same data (for the 95M variants), different supervision signals. This is precisely the setting where linear mode connectivity analysis is most informative, because it isolates the effect of the training objective on the loss landscape geometry.

---

## 10. What I Would Have Done Differently, and How I Would Extend This for a New Publication

### What I Would Have Changed

1. **Context length.** The 5-second limitation is the single largest bottleneck. I would have experimented with a curriculum strategy: pre-train at 5 seconds for the bulk of training, then extend to 15--30 seconds for a final phase, using gradient checkpointing to manage memory. Even a modest extension to 10 seconds (roughly 4 bars) would cover most phrase-level structures.

2. **Teacher refinement loop.** I would have invested in at least one round of iterative teacher refinement using MERT's own learned features to re-cluster and generate improved pseudo-labels. The paper dismisses this based on a negative result with MFCC features, but MFCC is a poor starting point; the conclusion does not generalise to EnCodec-derived features.

3. **A more principled evaluation aggregation.** Instead of averaging heterogeneous metrics, I would have used rank-based aggregation: for each task, rank all models, then report the mean rank. This avoids the incommensurability problem and is standard in multi-task benchmarking (e.g., GLUE, SUPERB).

4. **Include a fine-tuning evaluation.** Even a single fine-tuning result on 2--3 tasks would substantially increase the paper's practical impact and allow comparison with methods that only report fine-tuning numbers.

5. **Characterise the training data.** At minimum, report the genre distribution, language distribution (for vocals), and geographic origin distribution of the 160K-hour dataset. This is essential for understanding potential biases.

### How I Would Extend This Work

**Publication Idea: "Merging Music and Speech SSL Models in Weight Space for Universal Audio Understanding"**

The core premise: MERT and HuBERT (or data2vec) are architecturally identical Transformer encoders trained on different domains (music vs. speech) with different teacher signals. Can we merge them in weight space to produce a single model that performs well on both speech and music downstream tasks?

**Experimental Design:**

1. **Stage 1: Establish linear mode connectivity.** Take MERT-95M and HuBERT-base (both 95M, same architecture). Evaluate linear interpolations $\theta_{\text{merged}} = (1 - \lambda) \cdot \theta_{\text{MERT}} + \lambda \cdot \theta_{\text{HuBERT}}$ for $\lambda \in [0, 1]$ on both SUPERB (speech) and MARBLE (music) benchmarks. If the interpolated models degrade severely, apply permutation alignment first, then repeat.

2. **Stage 2: Apply advanced merging methods.** Compare linear interpolation, task arithmetic, TIES-Merging, and DARE with permutation alignment on the MERT/HuBERT pair. Evaluate whether the merged model retains performance on both speech ASR/speaker verification and music tagging/key detection.

3. **Stage 3: Merge MERT variants.** Merge MERT models trained with different teachers (K-means vs. RVQ-VAE, with vs. without CQT). This is a more controlled experiment (same domain, same data, different supervision) and tests whether teacher diversity creates complementary representations that can be combined.

4. **Stage 4: Extension to the SpeechMapper projector.** Use MERT as the encoder for a music-specific SpeechMapper projector, train task-specific variants (music tagging, transcription, music QA), merge them, and evaluate whether the merged projector achieves multi-task music understanding through a single LLM interface.

This research programme directly extends your existing expertise in permutation alignment and linear mode connectivity, applies it to a new and under-explored domain (music/audio SSL), and produces both a practical contribution (merged universal audio models) and theoretical insights (geometry of the loss landscape for multi-domain audio SSL).

---

*Analysis prepared March 2026. All claims about the paper reflect the content of arXiv:2306.00107v5 (ICLR 2024 camera-ready).*
