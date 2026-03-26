# Paper Analysis: Self-supervised Predictive Coding Models Encode Speaker and Phonetic Information in Orthogonal Subspaces

**Authors:** Oli Danyi Liu, Hao Tang, Sharon Goldwater  
**Affiliation:** School of Informatics, University of Edinburgh  
**Venue:** Interspeech 2023 (arXiv: 2305.12464v3)

---

## 1. High-Level Summary (The "TL;DR")

- The paper demonstrates that self-supervised predictive coding speech models (APC, CPC) encode speaker and phonetic information in **nearly orthogonal subspaces** of the representation space, discovered through PCA on aggregated frame-level representations.
- The authors exploit this geometric property to propose a **simple, transcription-free speaker normalization method**: collapse the speaker subspace by projecting representations onto its orthogonal complement, effectively erasing speaker identity while preserving phonetic content.
- The method **outperforms utterance-level standardization** baselines on ABX phone discrimination tasks, generalizes to unseen speakers, and is compatible with streaming applications because the projection directions are precomputed.

---

## 2. The Core Problem

Self-supervised speech representations are known to jointly encode multiple types of information: phonetic content, speaker identity, prosody, and more. Prior work has catalogued **what** information is encoded and **where** in the network hierarchy it resides (e.g., which layer), but essentially no work has examined **how** different information types are geometrically distributed across the dimensions of the representation space.

This is a meaningful gap for several reasons:

1. **Interpretability**: Understanding the geometric layout of information in learned representations provides a much richer picture than simple probing accuracy.
2. **Disentanglement**: If speaker and phonetic information occupy separate regions of the space, disentanglement becomes a trivial linear operation rather than requiring adversarial training or complex generative models.
3. **Practical utility**: Speaker variability is a persistent confound in unsupervised speech tasks (e.g., zero-resource phone discovery, acoustic word embedding). A lightweight, principled speaker normalization technique would be immediately useful.

Existing approaches to speaker normalization in SSL representations, such as utterance-level mean subtraction and variance normalization, are heuristic and lack a clear theoretical motivation tied to the structure of the representation space. The authors aim to fill this gap with a hypothesis-driven geometric analysis.

---

## 3. Key Contributions

1. **Empirical demonstration of orthogonal subspace structure**: The paper provides quantitative evidence that speaker and phonetic information reside in nearly orthogonal subspaces within the representations of predictive coding models (APC, CPC-small, CPC-big).
2. **A PCA-based analysis methodology**: A simple but effective pipeline consisting of aggregating frame-level representations by speaker, phone, and their joint combinations, followed by PCA, to identify and compare the principal directions encoding each source of variation.
3. **A novel speaker normalization technique**: Projecting representations onto the orthogonal complement of the speaker subspace, requiring only speaker labels (no transcriptions), and shown to generalize to unseen speakers.
4. **Empirical validation via probing and ABX discrimination**: Showing that collapsing the speaker subspace nearly eliminates speaker classification accuracy, improves phone discrimination (within- and across-speaker ABX), and outperforms the utterance-level standardization baseline from prior work.

---

## 4. Methodology & Technical Details

### 4.1 Representation Aggregation

The method begins by collecting frame-level hidden representations $\mathbf{z} \in \mathbb{R}^D$ from an SSL model, where $D$ is the hidden dimensionality (256 or 512 depending on the model). These are aggregated in three ways:

- **By speaker**: For each speaker $s \in S$, compute the mean over all frames:
$$M_{\text{spk}}[s] = \frac{1}{|Z_s|}\sum_{\mathbf{z} \in Z_s} \mathbf{z}$$
  This yields a $|S| \times D$ matrix $M_{\text{spk}}$.

- **By phone**: For each phone $p \in P$, compute the mean over all frames aligned to that phone:
$$M_{\text{phn}}[p] = \frac{1}{|Z_p|}\sum_{\mathbf{z} \in Z_p} \mathbf{z}$$
  This yields a $|P| \times D$ matrix $M_{\text{phn}}$.

- **By speaker-phone pair**: For each $(s, p)$ combination:
$$M_{\text{joint}}[s,p] = \frac{1}{|Z_{s,p}|}\sum_{\mathbf{z} \in Z_{s,p}} \mathbf{z}$$
  This yields a $|S||P| \times D$ matrix $M_{\text{joint}}$.

The aggregation step is critical. The authors note that running PCA on raw frame-level representations does not yield clear patterns, presumably because per-frame noise swamps the structured inter-speaker and inter-phone variance.

### 4.2 PCA-Based Subspace Identification

Standard PCA is applied independently to $M_{\text{spk}}$, $M_{\text{phn}}$, and $M_{\text{joint}}$, producing three sets of eigenvectors (principal directions) sorted by decreasing eigenvalue.

- The **speaker subspace** is spanned by the top $k$ eigenvectors of $M_{\text{spk}}$.
- The **phone subspace** is spanned by the top $k$ eigenvectors of $M_{\text{phn}}$.
- The **joint subspace** of $M_{\text{joint}}$ captures both sources of variation in a single decomposition.

To measure whether the speaker and phone subspaces are orthogonal, the authors compute pairwise similarities between the principal directions of $M_{\text{spk}}$ and $M_{\text{phn}}$ using the absolute value of the dot product:

$$\text{sim}(\mathbf{v}_i, \mathbf{u}_j) = |\mathbf{v}_i^\top \mathbf{u}_j| \in [0, 1]$$

A value near 0 indicates orthogonality; a value near 1 indicates alignment.

### 4.3 Speaker Subspace Collapsing (Speaker Normalization)

Given the orthogonality finding, the normalization procedure is straightforward. For a hidden vector $\mathbf{z}$ and a speaker principal direction $\mathbf{v}$, the component along $\mathbf{v}$ is removed:

$$\mathbf{z}' = \mathbf{z} - (\mathbf{z}^\top \mathbf{v})\mathbf{v}$$

This is applied sequentially for each of the top $k$ speaker principal directions, effectively projecting $\mathbf{z}$ onto the subspace orthogonal to the speaker subspace. In matrix form, if $V \in \mathbb{R}^{D \times k}$ contains the $k$ speaker directions as columns, the projection is:

$$\mathbf{z}' = \mathbf{z} - V V^\top \mathbf{z} = (I - V V^\top)\mathbf{z}$$

The number of dimensions $k$ to collapse is chosen either by a cumulative variance threshold (e.g., 95% of speaker variance explained) or by tuning on a development set (e.g., minimizing across-speaker ABX error).

Key design considerations:

- **No transcriptions required**: The speaker subspace is computed solely from speaker-labeled data, using only $M_{\text{spk}}$.
- **Generalization**: The speaker subspace is learned from a training set and applied to test sets with disjoint speakers.
- **Streaming compatible**: The projection matrix $(I - VV^\top)$ is precomputed; applying it to incoming frames is a single matrix-vector multiply.

### 4.4 Models Under Study

| Model | Architecture | Layers | Hidden Dim | Horizon $K$ | Training Data |
|-------|-------------|--------|-----------|------------|--------------|
| CPC-big | 5-layer CNN + 4-layer LSTM | Extract layer 2 | 512 | 12 | LibriLight 6k hrs |
| CPC-small | 5-layer CNN + 2-layer LSTM | Extract layer 2 | 256 | 12 | LibriSpeech 100 hrs |
| APC | Log Mel + 3-layer LSTM | Extract layer 3 | 512 | 3 | LibriSpeech 360 hrs |

Both CPC models use contrastive prediction (distinguishing the true future frame from negatives drawn **within the same speaker**). APC uses autoregressive prediction minimizing L2 distance to the target future frame. The within-speaker negative sampling in CPC is an important design detail, as it means the contrastive objective already encourages the model to ignore speaker identity when discriminating between positive and negative samples.

---

## 5. Experiments & Results

### 5.1 Datasets

- **Analysis**: LibriSpeech `dev-clean` (40 speakers: 19 male, 21 female, ~8 min each).
- **Speaker subspace training**: LibriSpeech `train-clean-100` (251 speakers, ~25 min each).
- **Evaluation**: LibriSpeech `dev-clean` and `test-clean` (40 speakers each). All speaker sets are disjoint.
- **Phone labels**: 39 phone categories obtained via forced alignment with a Kaldi acoustic model. Silence and spoken noise frames are excluded.

### 5.2 Baselines

- **Utterance-level centering**: Subtract the per-utterance mean from all frames in that utterance.
- **Utterance-level standardization**: Centering plus per-dimension variance rescaling.
- **Speaker-level variants** of the above (used when all speakers are known in advance).

### 5.3 Evaluation Metrics

- **Speaker probing accuracy** (linear classifier): Measures how much speaker identity can be decoded. Higher error = better normalization.
- **Phone probing accuracy** (linear classifier): Measures phonetic information retention. Error should remain low.
- **ABX phone discrimination error** (within- and across-speaker): Asks whether triphone $x$ is closer to same-type triphone $a$ than different-type triphone $b$. Lower error = better phonetic discriminability.

### 5.4 Key Quantitative Findings

**Orthogonality analysis (CPC-big on `dev-clean`)**:
Among the top 20 speaker directions, the average similarity (absolute dot product) with the most-aligned phone direction is only **0.13** (variance: 0.002, max: 0.26). This is strong evidence of near-orthogonality.

**Seen-speaker normalization** (`dev-clean`, collapsing all 40 speaker directions):

| Model | Setting | Original | Speaker-level Centered | Speaker Space Collapsed |
|-------|---------|----------|----------------------|------------------------|
| CPC-big | Speaker error (%) | 0.45 | 76.07 | **82.30** |
| CPC-big | ABX across (%) | 4.11 | 3.97 | **3.77** |
| CPC-small | Speaker error (%) | 10.21 | 67.08 | **84.69** |
| CPC-small | ABX across (%) | 8.10 | 7.38 | **6.73** |
| APC | Speaker error (%) | 15.47 | 83.02 | **85.47** |
| APC | ABX across (%) | 9.83 | 9.09 | **9.24** |

Speaker space collapsing consistently achieves the highest speaker error rates (i.e., most effective at removing speaker identity) and the best ABX discrimination for the CPC models.

**Unseen-speaker normalization** (subspace from `train-clean-100`, evaluated on `test-clean`):

| Model | ABX Within | ABX Across |
|-------|-----------|-----------|
| CPC-big (original) | 3.29 | 4.22 |
| CPC-big (utt. centered) | 3.27 | 4.11 |
| CPC-big (**collapsed**) | **3.10** | **4.01** |
| CPC-small (original) | 5.86 | 7.48 |
| CPC-small (**collapsed**) | **4.85** | **6.37** |

The method generalizes well to unseen speakers and consistently outperforms utterance-level centering on CPC models.

**Notable exception**: APC shows weaker gains from the collapsing method when generalizing to unseen speakers, with ABX scores slightly worse than utterance centering on the test set.

### 5.5 Ablation: Number of Collapsed Dimensions

Figure 3 in the paper shows that for CPC-big:
- When collapsing a subspace learned from `dev-clean` (40 speakers) and applying to the same speakers, using all 40 directions is beneficial.
- When generalizing from `train-clean-100` (251 speakers), performance peaks around 50-60 dimensions, after which lower principal components start to overfit to training speakers, causing ABX error to rise.

The chosen variance thresholds (98%, 95%, 95% for CPC-big, CPC-small, APC, corresponding to 57, 36, 30 dimensions) represent reasonable operating points.

### 5.6 Qualitative Analysis

Projection of $M_{\text{joint}}$ onto its top dimensions reveals interpretable structure:
- **Dimension 0**: Encodes a sonority gradient (vowels at one end, fricatives/affricates at the other), with no speaker differentiation.
- **Dimensions 1 and 2**: Differentiate between male and female speakers while maintaining some phonetic ordering within each cluster.
- **Dimension 12**: Captures inter-speaker variance (individual speaker identity), with substantially less phonetic structure.

---

## 6. Strengths

- **Clean, testable hypothesis**: The orthogonal subspace hypothesis is stated precisely and tested with appropriate quantitative measures (absolute dot products between principal directions). The paper does not overclaim.
- **Simplicity and elegance**: The entire pipeline (aggregation, PCA, projection) uses standard linear algebra, requiring no neural network training, no adversarial objectives, and no hyperparameter-heavy procedures.
- **No transcription dependency for normalization**: The speaker subspace is computed from speaker labels alone, making it applicable in low-resource or unsupervised settings where phone labels are unavailable.
- **Streaming compatibility**: The precomputed projection matrix makes this method practical for real-time applications, unlike utterance-level centering which requires a full utterance before computing the mean.
- **Honest reporting**: The authors note cases where their method does not clearly beat baselines (APC on unseen speakers), and provide useful ablations on the number of collapsed dimensions.
- **Interpretable visualizations**: The projection plots (Figure 2) are informative and support the claims with more than just numerical metrics.
- **Good experimental controls**: Using disjoint speaker sets for training and evaluation of the speaker subspace, and comparing to both centering and standardization baselines.

---

## 7. Weaknesses & Limitations

- **Restricted model family**: The paper only examines predictive coding models (APC, CPC). This is acknowledged by the authors, but it is a significant limitation because the most impactful SSL models in 2023 and beyond (HuBERT, wav2vec 2.0, WavLM, Whisper encoder) use masked prediction objectives with bidirectional context. The orthogonality finding may or may not extend to these architectures, and the paper provides no evidence either way.

- **Confound in CPC's negative sampling**: Both CPC models draw negative samples **within the same speaker**. This training signal explicitly pushes the model to learn representations that discriminate content, not speaker. The observed orthogonality may thus be an artifact of this particular training choice rather than a general property of self-supervised speech representations. A CPC model with cross-speaker negatives could exhibit very different geometry.

- **Small and homogeneous evaluation**: LibriSpeech is English read speech from audiobooks, a notably clean and controlled domain. The 40 speakers in `dev-clean` and `test-clean` are a small sample. It remains unclear whether the orthogonality holds for: (a) spontaneous/conversational speech, (b) noisy or far-field conditions, (c) languages other than English, (d) larger and more diverse speaker populations where speaker variation is higher-dimensional.

- **PCA as a subspace estimator**: PCA identifies directions of maximum variance, but variance and information are not synonymous. The speaker subspace captures directions of maximal inter-speaker variance, but speaker information could also be encoded in directions of low variance (e.g., subtle spectral tilts) or in non-linear manifolds. The linear, variance-maximizing assumption is not validated beyond the probing experiments.

- **Aggregation hides within-class structure**: Averaging all frames of a speaker or phone into a single vector discards substantial temporal and contextual variation. The claim of orthogonality is about the subspaces of speaker and phone **means**, not about the full distribution of frame-level representations. Two subspaces of means can be orthogonal even if the raw frame-level distributions overlap substantially.

- **Phone labels required for analysis (though not normalization)**: The orthogonality analysis and phone probing both require forced-aligned phone transcriptions, which limits the self-contained unsupervised narrative of the paper.

- **Modest ABX improvements**: The absolute ABX error improvements from collapsing over centering are relatively small (often 0.1 to 0.5 percentage points). While consistent, these are modest enough that they may not translate to meaningful downstream gains in, e.g., unsupervised speech recognition or spoken term discovery.

- **No comparison to learned disentanglement**: The paper does not compare against neural approaches to speaker-phonetic disentanglement (e.g., VQ-VAE with factorized codebooks, adversarial speaker removal, or information bottleneck methods). This makes it difficult to assess whether the simplicity of PCA-based collapsing comes at a performance cost.

- **Missing analysis of what is removed**: The paper shows that speaker probing accuracy drops, but does not analyze whether other useful information (e.g., prosodic variation, speaking rate, emphasis) is collaterally removed along with speaker identity.

---

## 8. Practical Implications & Applications

- **Zero-resource speech processing**: In settings where no transcriptions are available, this method provides a principled, lightweight way to make SSL representations more phone-discriminative. It could plug directly into unsupervised phone discovery, spoken term detection, or acoustic word embedding pipelines.

- **Speaker-invariant features for ASR**: As a preprocessing step before fine-tuning SSL models on downstream ASR tasks, collapsing the speaker subspace may reduce the amount of speaker adaptation data needed, particularly for low-resource languages.

- **Speaker verification and anti-spoofing**: Understanding that speaker information lives in a particular subspace could also be exploited in the reverse direction: **extracting** the speaker subspace for speaker verification, rather than collapsing it.

- **Privacy-preserving speech processing**: Removing speaker identity at the representation level, before passing features to downstream models, could support privacy-sensitive deployment (e.g., voice assistants that process queries without retaining speaker-identifiable features).

- **Cognitive modeling**: The finding that predictive coding models implicitly disentangle speaker and phonetic information into orthogonal subspaces may serve as a testable prediction for neural representation geometry in human auditory cortex.

- **Streaming applications**: The precomputed projection matrix makes the method viable for real-time systems, unlike utterance-level normalization which requires buffering.

---

## 9. Future Work & Open Questions

### General Research Directions

1. **Extension to masked prediction models**: The most pressing question is whether the orthogonality property holds for HuBERT, wav2vec 2.0, WavLM, and other bidirectional SSL models. These models use fundamentally different training objectives (masked prediction of quantized targets with bidirectional context) and have substantially different representation geometries. Given that these are the dominant models in the field, the practical relevance of the current findings hinges on this extension.

2. **Beyond speaker vs. phone**: The subspace analysis could be extended to other factors of variation such as language, emotion, accent, speaking style, and recording channel. Whether these factors also occupy distinct subspaces, and whether they are mutually orthogonal, is an open empirical question.

3. **Non-linear extensions**: The orthogonality claim is inherently linear. Kernel PCA, autoencoders, or manifold learning methods could probe whether the linear subspace assumption captures most of the structure or whether significant non-linear entanglement exists.

4. **Cross-lingual and cross-domain generalization**: Does a speaker subspace learned from English LibriSpeech generalize to other languages or to spontaneous/noisy speech? This is critical for practical deployment.

5. **Theoretical explanation**: The paper documents the orthogonality but does not explain *why* predictive coding objectives produce it. A theoretical account, perhaps connected to the independence of speaker and phone generative factors, would strengthen the contribution.

### Implications for Model Merging in Speech Foundation Models

This paper is directly relevant to research on model merging for speech and audio foundation models in several ways:

- **Subspace structure as a guide for merging**: If task-specific fine-tuned speech models encode distinct task knowledge in identifiable subspaces, the same PCA-based analysis could be used to understand what each model has learned and how they differ. This could inform whether task arithmetic, TIES-Merging, or permutation alignment is the appropriate merging strategy.

- **Permutation alignment in the projector**: The finding that information is organized along specific principal directions within LSTM-based SSL models suggests that the alignment problem in model merging (identifying which neurons correspond across independently trained models) could be partially solved by aligning principal component directions rather than individual neurons. This is particularly relevant for the Transformer encoder layers within SpeechMapper-style projectors, where permutation symmetries are well-characterized.

- **Collateral damage during merging**: If task-specific projectors trained for ASR versus speech translation have different speaker subspace structures (e.g., because translation may need to preserve some speaker characteristics for voice transfer), naive merging could distort one model's speaker-phone geometry. Understanding and preserving orthogonal subspace structure during merging could be a design criterion.

- **Diagnostic tool for merged models**: After merging task-specific projectors, one could use this PCA-based analysis to verify that the merged model retains clean orthogonal structure. Degradation of orthogonality could serve as an early warning that merging has produced destructive interference.

- **Extension to other modalities**: The methodology is general enough to apply to music or environmental audio foundation models. The question of whether content vs. timbre, or event type vs. recording condition, are encoded orthogonally in models like AudioMAE or CLAP would be a natural extension.

---

## 10. What I Would Have Done Differently and How to Extend This Work

### Methodological Improvements

**Include masked prediction models from the start.** The paper's restriction to APC and CPC is understandable given the "predictive coding" framing, but scientifically, the most interesting question is whether the orthogonality is a general property of SSL speech representations or specific to unidirectional predictive coding. I would have included at least wav2vec 2.0 and HuBERT in the main experiments, even at the cost of reducing the depth of analysis per model. A negative result (non-orthogonality in masked prediction models) would be equally publishable and arguably more informative.

**Control for the CPC negative sampling confound.** The within-speaker negative sampling in CPC creates an explicit inductive bias toward speaker-invariant content representations. I would have trained or sourced a CPC model with unrestricted (cross-speaker) negative sampling to test whether orthogonality survives. Without this control, the causal attribution is ambiguous.

**Analyze higher layers and multi-layer representations.** The paper extracts from a single layer per model. Modern SSL model analysis (e.g., the SUPERB and LeBenchmark efforts) has shown that different layers encode different information. Analyzing the orthogonality across layers could reveal whether it emerges gradually or appears suddenly.

**Quantify subspace overlap more rigorously.** Beyond pairwise dot products between individual eigenvectors, I would use the *principal angle* between subspaces (computed via SVD of the product of the projection matrices) as a more geometrically principled measure. The principal angle between two $k$-dimensional subspaces provides a single scalar summary of their relative orientation, and has well-studied statistical properties.

Formally, given orthonormal bases $U \in \mathbb{R}^{D \times k_1}$ (speaker subspace) and $W \in \mathbb{R}^{D \times k_2}$ (phone subspace), the principal angles $\theta_1 \leq \cdots \leq \theta_{\min(k_1, k_2)}$ are defined by $\cos(\theta_i) = \sigma_i(U^\top W)$, where $\sigma_i$ denotes the $i$-th singular value. If all principal angles are near $\pi/2$, the subspaces are truly orthogonal.

### Extension for a New Publication

I would propose a paper titled something like: **"Orthogonal Information Geometry in Self-Supervised Speech Models: From Predictive Coding to Masked Prediction and Beyond."**

The core contribution would be:

1. **Comprehensive comparison across SSL model families**: Apply the PCA-based orthogonality analysis to predictive coding (APC, CPC), masked prediction (HuBERT, wav2vec 2.0, WavLM), and encoder-decoder (Whisper) models. Analyze all layers, not just the final one. This gives a taxonomy of which architectures and objectives produce orthogonal speaker-phone geometry.

2. **Expand the set of factors**: Beyond speaker and phone, analyze orthogonality with respect to language (in multilingual models), emotion, and noise/channel condition. This requires suitable multilingual and multi-condition datasets (e.g., VoxCeleb for speaker diversity, IEMOCAP for emotion, CHiME for noise).

3. **Connect to model merging**: Use the orthogonality analysis as a diagnostic and design tool for merging task-specific speech projectors. Concretely:
   - Train task-specific SpeechMapper projectors (ASR, ST, SQA) from a shared Stage 1 checkpoint.
   - Measure the subspace geometry of each task-specific projector.
   - Test whether preserving orthogonal structure during merging (e.g., via permutation alignment that respects principal directions) improves merged model performance compared to naive merging.

4. **Theoretical grounding**: Provide a probabilistic argument for why training objectives that depend on content discrimination within a speaker (or within a context window) should produce orthogonal factor representations, drawing on the independence of the generative factors and the information-theoretic properties of the training loss.

5. **Practical speaker normalization at scale**: Evaluate the collapsing method on a modern, large-scale SSL model (e.g., WavLM-Large) across multiple downstream tasks beyond ABX, including actual ASR word error rate, speaker verification EER, and emotion recognition accuracy, to demonstrate practical utility at a level that would be convincing to the applied community.

This would constitute a substantial follow-up that bridges representational analysis, theoretical understanding, and practical application in the merging paradigm.
