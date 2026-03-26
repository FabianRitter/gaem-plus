# CLAUDE.md — GAEM+ Research Matrices Agent Reference

## Purpose

This file instructs any AI agent on how to interact with the **GAEM_Research_Matrices.xlsx** workbook located in the same directory. The workbook is Fabian Ritter's single source of truth for organizing his PhD research on **GAEM+ (Generalized Audio Encoder Merging)** — a framework for merging speech, music, and audio encoders using orthogonal alignment, low-rank decomposition, and pre-training orthogonalization.

---

## Research Context (Read This First)

Fabian is a 3rd-year PhD student working on efficient machine learning for speech processing. His core research contribution is **GAEM+**, which combines:

1. **Orthogonal alignment** (from GLMC) — richer than permutation-only alignment
2. **Low-rank + sparse task vector decomposition** (from LoRS-Merging) — structure-aware merging
3. **Pre-training orthogonalization** (from OSRM) — subspace interference prevention
4. **Audio LLM integration** — replacing dual encoders in systems like SALMONN with a single merged encoder

The key models being merged are **HuBERT** (speech), **MERT** (music), and **BEATs** (audio events). The downstream target is integration into Speech/Audio LLMs (primarily SALMONN).

Fabian has two prior published papers:
- **Paper 1**: Permutation-based alignment for audio encoder merging (Git Re-Basin applied to audio)
- **Paper 2**: Task Arithmetic Distillation for audio encoders

### Critical Framing: Why Encoder Merging, Not Just ICL?

GAEM+ operates at the **encoder level** (audio → representations), NOT the LLM level. This distinction is essential for any agent writing about, evaluating, or positioning this research:

1. **ICL and merging are complementary, not competing.** In-context learning adapts what the LLM *does* with representations. Encoder merging improves the *quality* of representations the LLM receives. No amount of in-context examples can recover information discarded at the encoding stage (e.g., Whisper discards music harmonics; no ICL fixes that).

2. **Audio ICL is inherently limited.** Each audio in-context example consumes hundreds of LLM tokens, so practical ICL is limited to 1–3 audio examples. This makes encoder representation quality the dominant factor for audio LLM performance.

3. **The architectural efficiency argument.** Current systems use dual encoders (SALMONN: Whisper + BEATs ≈ 1.2B params) or accept limited coverage (Qwen-Audio: Whisper-only). GAEM+ offers a single merged encoder with multi-domain coverage at roughly single-encoder cost.

4. **Extensibility without retraining.** New audio domains can be merged in without retraining the LLM or adapter — unlike ICL (needs examples every call) or fine-tuning (needs full pipeline retraining).

**When the agent writes about GAEM+, always frame it as improving the foundation on which ICL operates, never as replacing ICL.**

---

## Workbook Structure

**File**: `GAEM_Research_Matrices.xlsx`

### Sheet 1: "Model Merging Papers"
The primary literature tracking matrix. 72+ papers, 23 columns (A–W), header in row 1, data from row 2 onward. Frozen panes at C2. Auto-filter enabled.

| Column | Header | Description | Validation (dropdown) |
|--------|--------|-------------|-----------------------|
| A | ID | Sequential integer | Free |
| B | Paper Title | Full paper title | Free |
| C | First Author | Last name of first author | Free |
| D | Year | Publication year (integer) | Free |
| E | Venue/Status | Conference, journal, or "arXiv YYYY" | Free |
| F | Merging Technique Category | Primary technique class | `Weight Averaging`, `Task Arithmetic`, `Fisher Merging`, `TIES-Merging`, `DARE`, `Permutation Alignment`, `Orthogonal Alignment`, `Low-Rank Decomposition`, `Sparse Merging`, `LoRA Merging`, `Layer Swapping`, `Knowledge Editing`, `Adaptive/Learned`, `MoE-Based`, `Distillation-Based`, `SVD-Based`, `Other` |
| G | Symmetry Type Addressed | What symmetries the paper exploits | `None`, `Permutation`, `Semi-Permutation`, `Orthogonal`, `Invertible`, `Multiple`, `N/A` |
| H | Core Innovation | One-sentence description of the key contribution | Free |
| I | Modality | Domain the paper operates in | `NLP`, `Vision`, `Speech`, `Audio`, `Music`, `Multimodal`, `Speech+Audio`, `General/Theory` |
| J | Base Model(s) | Models used in experiments | Free |
| K | Datasets / Benchmarks | Evaluation datasets | Free |
| L | Multi-Task or Multi-Domain? | Whether the paper addresses multiple tasks | `Yes`, `No`, `Partial`, `N/A` |
| M | Training-Free? | Whether the merging requires additional training | `Yes`, `No`, `Partial`, `N/A` |
| N | Handles Different Widths? | Whether it can merge models of different hidden dimensions | `Yes`, `No`, `Partial`, `N/A` |
| O | LoRA / Adapter Based? | Whether the method is LoRA/adapter-specific | `Yes`, `No`, `Partial`, `N/A` |
| P | Interference Mitigation Strategy | How the paper handles task/domain interference | Free |
| Q | Relevance to GAEM+ | How important to our research | `Critical`, `High`, `Medium`, `Low` |
| R | Connection to Our Work | Specific way this paper connects to GAEM+ | Free |
| S | Key Limitations | Main weaknesses or gaps | Free |
| T | Key Takeaway for Us | Actionable insight for our pipeline | Free |
| U | Reading Status | Current reading progress | `Not Read`, `Skimmed`, `In Progress`, `Read`, `Deep Read` |
| V | Priority | Reading priority | `Must Read`, `Should Read`, `Nice to Have`, `Reference Only` |
| W | ArXiv / DOI Link | URL or arXiv ID | Free |

### Sheet 2: "Speech-Audio LLM Frameworks"
Tracks Speech/Audio LLM systems for GAEM+ integration evaluation. Header in row 1, data from row 2. 22 columns (A–V).

| Column | Header | Description | Validation |
|--------|--------|-------------|------------|
| A | ID | Sequential integer | Free |
| B | Framework Name | Name of the Speech/Audio LLM | Free |
| C | First Author | Last name of first author | Free |
| D | Year | Publication year | Free |
| E | Venue | Conference or arXiv | Free |
| F | Speech/Audio Encoder(s) | Which encoder(s) are used | Free |
| G | Encoder Architecture | Single, dual, MoE, etc. | Free |
| H | LLM Backbone | Which LLM is used | Free |
| I | Adapter/Bridge Type | How encoder connects to LLM | Free |
| J | Tasks Supported | List of audio tasks | Free |
| K | Training Strategy | How the system is trained | Free |
| L | Multi-Encoder? | Uses multiple audio encoders? | Free |
| M | Encoder Frozen? | Is the encoder frozen during training? | Free |
| N | Token Compression | How audio tokens are compressed | Free |
| O | Open Source? | Code/weights availability | Free |
| P | ICL Capability | How well the framework supports in-context learning for audio tasks. **Key for GAEM+ justification**: frameworks with stronger ICL but encoder bottlenecks are the best targets, because GAEM+ improvements are clearly attributable to better representations, not confounded with weak ICL. | `Strong`, `Moderate`, `Weak`, `Unknown`, `N/A` |
| Q | Merging Compatibility | How easily the encoder can be swapped with a merged one. `High` = modular encoder with clean interface. `Low` = encoder deeply entangled with architecture. | `High`, `Medium`, `Low`, `Unknown` |
| R | Key Strengths | Main advantages of this framework | Free |
| S | Key Limitations | Main weaknesses or gaps | Free |
| T | GAEM+ Integration Feasibility | How hard it is to integrate GAEM+. `Easy` = drop-in replacement. `Hard` = significant re-engineering. | `Easy`, `Moderate`, `Hard`, `Unclear` |
| U | Could Replace Encoder With Merged? | Whether a merged encoder could serve as drop-in | Free |
| V | Notes for Our Work | Specific plan for GAEM+ integration | Free |

### Sheet 3: "Decision Matrix"
Scored comparison of Speech LLM frameworks (1–5 scale, 8 criteria). Row 3 = headers, row 4+ = data. Column J has `=SUM(B:I)` formulas.

### Sheet 4: "Research Pipeline"
Milestone tracker. Row 3 = headers, row 4+ = tasks. Status column uses: `Not Started`, `In Progress`, `Blocked`, `Done`.

### Sheet 5: "Reading Priority"
Weekly reading queue. Row 4 = headers, row 5+ = entries. Derived from Sheet 1 priorities.

### Sheet 6: "Workflow Guide"
Static reference. Do not modify unless Fabian explicitly asks.

---

## Agent Operations

### Adding a New Model Merging Paper

When Fabian reads or mentions a new paper that should be tracked:

1. **Determine the next ID**: Find the last used ID in column A of "Model Merging Papers" and increment by 1.

2. **Fill ALL 23 columns** using this checklist:
   - Columns A–E: Bibliographic info (ID, title, first author, year, venue)
   - Column F: Classify the merging technique. If it combines multiple, use the primary one and note others in column H.
   - Column G: Identify the symmetry type. Use `None` if the paper doesn't address symmetries. Use `Multiple` if it addresses more than one.
   - Column H: Write ONE sentence capturing the core novelty. Be specific, not generic.
   - Column I: Classify modality. Use `Speech+Audio` if it covers both. Use `General/Theory` for theory papers.
   - Columns J–K: List concrete model names and dataset names.
   - Columns L–O: Answer with the constrained dropdown values.
   - Column P: Describe the interference strategy in 5–15 words. Use "None" or "N/A" if the paper doesn't address interference.
   - Column Q: Rate relevance by asking: "Does this paper provide a technique, insight, or baseline that directly affects the GAEM+ pipeline?" Critical = core component; High = important baseline or insight; Medium = useful context; Low = tangential.
   - Column R: Be SPECIFIC. Don't write "relevant to our work." Write exactly HOW it connects (e.g., "Use their SVD decomposition before our orthogonal alignment step").
   - Column S: Identify the gap or weakness from GAEM+'s perspective.
   - Column T: What should Fabian DO with this paper's insight? Actionable, not descriptive.
   - Columns U–V: Set reading status and priority.
   - Column W: Include arXiv ID or DOI if available.

3. **Cross-update other sheets if needed**:
   - If the paper is `Must Read` or `Should Read`, add it to "Reading Priority" (Sheet 5).
   - If the paper introduces a new Speech/Audio LLM framework, also add a row to Sheet 2.
   - If the paper affects the research pipeline, note it in the "Papers to Reference" column of Sheet 4.

### Adding a New Speech/Audio LLM Framework

1. Increment ID in Sheet 2.
2. Fill all 22 columns (A–V). Pay special attention to:
   - Column P (ICL Capability): Rate how well the framework supports in-context learning. `Strong` = demonstrated few-shot audio ICL. `Moderate` = LLM backbone supports ICL but limited audio ICL evidence. `Weak` = no ICL capability or very small LLM. Frameworks with stronger ICL but encoder bottlenecks are the BEST GAEM+ targets (improvements clearly come from better representations).
   - Column Q (Merging Compatibility): How easily can the encoder be swapped? `High` = modular encoder, clean interface. `Low` = encoder deeply entangled with architecture.
   - Column T (GAEM+ Integration Feasibility): `Easy` = drop-in replacement possible. `Hard` = significant re-engineering needed.
   - Column V (Notes for Our Work): Specific plan for how we'd integrate GAEM+.
3. Add the framework to Sheet 3 (Decision Matrix) with scores across all 8 criteria.

### Updating Reading Status

When Fabian finishes reading a paper:
1. Update column U in Sheet 1 (e.g., `Not Read` → `Read`).
2. Fill in or refine columns H, P, R, S, T if they were previously empty or vague.
3. Update Sheet 5 if the paper was listed there.

### Updating the Research Pipeline

When a milestone changes status:
1. Update column C in Sheet 4.
2. If a task is blocked, note the blocker in column F (Notes).
3. If new tasks emerge from a paper reading, add rows in the appropriate phase.

---

## Quality Rules

1. **No empty "Connection" cells**: Column R (Connection to Our Work) must NEVER be left blank for any paper with relevance `High` or `Critical`. If unsure, write a hypothesis.

2. **Dropdown values only**: For validated columns (F, G, I, L–O, Q, U, V in Sheet 1; P, Q, T in Sheet 2), use ONLY the allowed values listed above. Using a non-listed value will break the dropdown validation.

3. **One technique per cell in column F**: If a paper uses multiple techniques, pick the PRIMARY one for column F and describe the combination in column H (Core Innovation).

4. **Relevance is GAEM+-centric**: Always rate relevance from the perspective of "How much does this help build or validate the GAEM+ pipeline?" Not general academic importance.

5. **Actionable takeaways**: Column T should answer "What should we DO?" not "What did they find?" Bad: "They found orthogonal alignment works." Good: "Apply their Procrustes solver to HuBERT-MERT alignment in Experiment 1."

6. **Preserve formatting**: The workbook uses specific styling (navy headers, data validations, frozen panes, auto-filters). When editing with openpyxl, load with `load_workbook()` (NOT `data_only=True`) to preserve formulas and formatting.

---

## Code Template for Adding a Paper

```python
from openpyxl import load_workbook

XLSX_PATH = "GAEM_Research_Matrices.xlsx"  # adjust path as needed

def add_merging_paper(paper_data: dict):
    """
    paper_data keys must match:
      title, first_author, year, venue, technique, symmetry,
      innovation, modality, models, datasets, multi_task,
      training_free, diff_widths, lora_based, interference,
      relevance, connection, limitations, takeaway,
      status, priority, link
    """
    wb = load_workbook(XLSX_PATH)
    ws = wb["Model Merging Papers"]
    next_id = ws.max_row  # row count minus header = last ID, so next = max_row
    row = [
        next_id,
        paper_data["title"],
        paper_data["first_author"],
        paper_data["year"],
        paper_data["venue"],
        paper_data["technique"],
        paper_data["symmetry"],
        paper_data["innovation"],
        paper_data["modality"],
        paper_data["models"],
        paper_data["datasets"],
        paper_data["multi_task"],
        paper_data["training_free"],
        paper_data["diff_widths"],
        paper_data["lora_based"],
        paper_data["interference"],
        paper_data["relevance"],
        paper_data["connection"],
        paper_data["limitations"],
        paper_data["takeaway"],
        paper_data["status"],
        paper_data["priority"],
        paper_data["link"],
    ]
    ws.append(row)
    # Apply body styling to the new row
    from openpyxl.styles import Font, Alignment, Border, Side
    body_font = Font(name='Arial', size=10)
    wrap = Alignment(wrap_text=True, vertical='top')
    border = Border(
        left=Side(style='thin', color='D0D0D0'),
        right=Side(style='thin', color='D0D0D0'),
        top=Side(style='thin', color='D0D0D0'),
        bottom=Side(style='thin', color='D0D0D0')
    )
    for cell in ws[ws.max_row]:
        cell.font = body_font
        cell.alignment = wrap
        cell.border = border
    wb.save(XLSX_PATH)


def add_speech_llm(framework_data: dict):
    """
    framework_data keys (22 columns, A–V):
      name, first_author, year, venue, encoders, encoder_arch,
      llm_backbone, adapter, tasks, training_strategy,
      multi_encoder, encoder_frozen, token_compression,
      open_source, icl_capability, merging_compat, strengths,
      limitations, feasibility, replace_encoder, notes
    """
    wb = load_workbook(XLSX_PATH)
    ws = wb["Speech-Audio LLM Frameworks"]
    next_id = ws.max_row
    row = [
        next_id,
        framework_data["name"],
        framework_data["first_author"],
        framework_data["year"],
        framework_data["venue"],
        framework_data["encoders"],
        framework_data["encoder_arch"],
        framework_data["llm_backbone"],
        framework_data["adapter"],
        framework_data["tasks"],
        framework_data["training_strategy"],
        framework_data["multi_encoder"],
        framework_data["encoder_frozen"],
        framework_data["token_compression"],
        framework_data["open_source"],
        framework_data["icl_capability"],
        framework_data["merging_compat"],
        framework_data["strengths"],
        framework_data["limitations"],
        framework_data["feasibility"],
        framework_data["replace_encoder"],
        framework_data["notes"],
    ]
    ws.append(row)
    from openpyxl.styles import Font, Alignment, Border, Side
    body_font = Font(name='Arial', size=10)
    wrap = Alignment(wrap_text=True, vertical='top')
    border = Border(
        left=Side(style='thin', color='D0D0D0'),
        right=Side(style='thin', color='D0D0D0'),
        top=Side(style='thin', color='D0D0D0'),
        bottom=Side(style='thin', color='D0D0D0')
    )
    for cell in ws[ws.max_row]:
        cell.font = body_font
        cell.alignment = wrap
        cell.border = border
    wb.save(XLSX_PATH)
```

---

## Common Agent Scenarios

### "I just read paper X, add it to the tracker"
1. Ask Fabian for any missing details (or infer from the paper if you have access).
2. Call `add_merging_paper()` with all 23 fields filled.
3. If `Must Read` or `Should Read`, also add to Reading Priority sheet.
4. Confirm what was added and ask if the relevance/connection assessments look right.

### "Update paper X — I finished reading it"
1. Find the row by title match in Sheet 1.
2. Update column U (Reading Status).
3. Ask Fabian to refine columns R (Connection) and T (Takeaway) if they were placeholders.

### "I found a new Speech LLM framework"
1. Call `add_speech_llm()` with all 22 fields (including `icl_capability`).
2. Add a row to the Decision Matrix (Sheet 3) with initial scores.
3. Ask Fabian to validate the ICL Capability, Merging Compatibility, and GAEM+ Feasibility ratings.

### "What should I read next?"
1. Open Sheet 5 (Reading Priority) and Sheet 1.
2. Filter Sheet 1 for `Not Read` or `Skimmed` papers with priority `Must Read` or `Should Read`.
3. Cross-reference with Sheet 4 (Research Pipeline) — recommend papers relevant to the current phase.
4. Suggest 2–3 papers with rationale tied to the active milestone.

### "How does paper X relate to paper Y?"
1. Look up both papers in Sheet 1.
2. Compare columns F (Technique), G (Symmetry), P (Interference), R (Connection).
3. Synthesize a comparison focused on implications for GAEM+.

### "What gaps exist in my literature coverage?"
1. Analyze Sheet 1 by grouping column F (Technique) and column I (Modality).
2. Identify technique categories with few or no Speech/Audio papers.
3. Cross-reference with GAEM+ components (orthogonal alignment, LoRS decomposition, OSRM pre-training) to find under-covered areas.
4. Report gaps with suggested search queries.

---

## Deep Reading Notes System (`paper-readings/`)

### Purpose

The Excel workbook captures *structured metadata* (23 columns per paper). The `paper-readings/` folder holds *deep analysis* — full reading notes, methodology breakdowns, connection analysis, and research ideas sparked by each paper. The two systems are linked by **Excel Row ID** embedded in every .md filename.

### Folder Structure

```
paper-readings/
├── _template.md              ← Copy this for every new paper
├── model-merging/            ← Merging techniques, alignment, interference
├── speech-audio-llms/        ← Speech/Audio LLM frameworks, adapters, projectors
├── audio-encoders/           ← SSL models (HuBERT, MERT, BEATs), encoder analysis
└── theory-foundations/        ← Symmetry theory, LMC analysis, general ML theory
```

### File Naming Convention

```
{EXCEL_ID}_{short-kebab-title}.md
```

- `EXCEL_ID` = the ID from Sheet 1 column A (or `S2-{ID}` for Sheet 2-only entries)
- `short-kebab-title` = 1-3 word kebab-case identifier
- Examples: `05_glmc.md`, `06_lors-merging.md`, `73_speechmapper.md`, `S2-01_salmonn.md`

### Folder Assignment Rules

| Folder | When to use | Examples |
|--------|------------|---------|
| `model-merging/` | Paper's primary contribution is a merging technique, alignment method, or interference mitigation | Git Re-Basin, GLMC, TIES, DARE, LoRS, OSRM, AdaMerging |
| `speech-audio-llms/` | Paper presents or analyzes a Speech/Audio LLM framework, adapter, or projector | SALMONN, Qwen-Audio, SpeechMapper, WavLLM, MoWE-Audio |
| `audio-encoders/` | Paper focuses on audio SSL models, encoder architecture, or audio representation analysis | HuBERT analysis, MERT, BEATs, wav2vec, Dynamic-SUPERB |
| `theory-foundations/` | Paper is primarily theoretical (symmetry, LMC, generalization) or a survey | Entezari 2022, symmetry theory, merging surveys |

If a paper spans two categories, place it in the folder matching its **primary contribution** and note the cross-reference in the .md file header.

### Template Structure

The template (`_template.md`) has 7 sections. Each section includes `[EXCEL →]` markers showing which Excel column(s) it feeds:

| Template Section | Purpose | Feeds Excel Column(s) |
|-----------------|---------|----------------------|
| 1. TL;DR | 2-3 bullet summary | H (Core Innovation) |
| 2. Core Problem | What gap the paper addresses | — (context only) |
| 3. Method | Technical details for re-implementation | J (Models), K (Datasets) |
| 4. Key Results | Concrete numbers and benchmarks | J, K |
| 5. Strengths & Limitations | Balanced assessment | S (Key Limitations) |
| 6. Connection to GAEM+ | **Most important section** | R (Connection), T (Takeaway) |
| 7. Ideas Sparked | New experiments, extensions | Sheet 4 (Pipeline), proposal docs |

### Agent Workflow: Processing a New Reading Note

When Fabian provides a completed .md reading note (or asks the agent to create one from a paper):

1. **Assign ID**: Check Sheet 1 for the next available ID. If the paper already has a row (added earlier as `Not Read`), use that existing ID.

2. **Place the file**: Copy/save as `{ID}_{kebab-title}.md` in the correct subfolder.

3. **Extract Excel updates from the .md**: Read the completed note and update the Excel row:
   - Section 1 (TL;DR) → Column H (Core Innovation): condense to ONE sentence
   - Section 3 (Method) → Columns J-K: extract model names and datasets
   - Section 5 (Limitations) → Column S: top 2-3 limitations in one cell
   - Section 6 (Connection to GAEM+) → Column R: 1-2 sentence connection statement
   - Section 6 ("What we should DO") → Column T: single most actionable takeaway
   - Update Column U (Reading Status) to match the note's stated depth
   - If the note reveals the paper is more/less relevant than initially estimated, update Column Q

4. **Cross-update other sheets**:
   - If Section 7 (Ideas Sparked) suggests new experiments → add tasks to Sheet 4 (Pipeline)
   - If the paper is a Speech/Audio LLM framework → add/update row in Sheet 2
   - If the note changes reading priority → update Sheet 5

5. **Confirm with Fabian**: Show the extracted Excel updates and ask if the Connection (R) and Takeaway (T) assessments are correct before writing.

### Agent Workflow: Creating a Reading Note from Scratch

When Fabian says "I'm reading paper X, help me analyze it":

1. **Search for the paper** (web search or provided PDF/link).
2. **Copy `_template.md`** to the appropriate subfolder with the correct filename.
3. **Fill all 7 sections**, paying special attention to Section 6 (Connection to GAEM+). Use the GAEM+ context from this CLAUDE.md to write specific, actionable connections — not generic observations.
4. **Run the Excel update workflow** (steps 1-5 above).

### Quality Rules for Reading Notes

1. **Section 6 is mandatory and must be specific.** Never write "this is relevant to our work." Write exactly how: which GAEM+ component, which experiment, what to implement or test.

2. **Section 7 captures fleeting ideas.** Even half-formed ideas belong here. They get lost otherwise. These feed into the proposal docs and Pipeline tracker.

3. **One file per paper.** No multi-paper summary files. Cross-references go in Section 6 as links to other .md files (e.g., "see also `05_glmc.md`").

4. **The .md is the deep record; the Excel is the queryable index.** If you can only update one, update the Excel. But the .md is where the real understanding lives.

---

## File Paths

| Item | Path |
|------|------|
| This reference | `CLAUDE.md` (same directory as the Excel file) |
| Excel workbook | `GAEM_Research_Matrices.xlsx` |
| Reading notes folder | `paper-readings/` |
| Reading note template | `paper-readings/_template.md` |
| GAEM+ proposal (detailed) | `paper-merging-idea/Revised_Research_Analysis 2.md` |
| GAEM+ technical supplement | `paper-merging-idea/Technical_Supplement_GAEM 2.md` |
| Research proposal PPTX | `paper-merging-idea/GAEM_Plus_Research_Proposal-updated.pptx` |
| PhD research plan | `paper-merging-idea/PhD_Research_Plan_Fabian_Ritter 2.docx` |
| Model merging lit review slides | `../Literature Review Model Merging.pptx` |
| Speech LLM lit review slides | `../LLM-SPEECH-LLM-LIT-REVIEW/LLM-SPEECH LLM LITERATURE REVIEW.pptx` |
| Bibliography (all papers) | Uploaded as `merging-biib.rtf` |

All paths are relative to `Unified Model Merging Last PhD Work 2026/` unless prefixed with `../`.
