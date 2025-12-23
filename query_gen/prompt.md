# Query Generation from Transcript + Real Comments (Realism-Forced, Failure-Aware)

You generate **realistic, human-like search queries** for evaluating a YouTube-video retrieval system.

Priority: **real user behavior under confusion**, not clean textbook questions.

These queries must reflect how people actually search when they:
- are stuck on setup/errors
- half-understand terminology
- want next steps (deploy, save/load, hardware)
- ask comparisons (RAG vs fine-tuning)
- write fragments, typos, or non-native English

---

## Inputs

You will be given:
1) **Video Title**
2) **Transcript** (NO timestamps)
3) **Real YouTube Comments** (some are praise; some contain real questions, errors, and requests)

---

## Step 0 — Extract the “Comment Prior” (do this silently)

From the comments, infer common query motifs and phrasing patterns. Examples you should emulate:
- environment/setup pain (Colab/Linux/MPS/CUDA, `accelerate`, versions)
- training/debugging failures (NaN loss, memory errors, slow epochs)
- implementation details (save/load model, tokenizer mismatch, pad token)
- conceptual confusion (“is BERT an LLM?”, how LoRA/PEFT works)
- tradeoffs (RAG vs fine-tuning, transfer learning vs LoRA, combining methods)
- deployment/hardware (API/UI deployment, edge constraints, GPU memory)

Use this as a style and intent prior: queries should feel like they came from real commenters.

---

## Query Types (Fixed)

Generate queries in **three categories**:

### 1) Exact / Factual
Definitions, specific facts, “what does X mean”, “where is X set”.

### 2) Conceptual
Why/how/when, tradeoffs, interpretation, “should I do X or Y”.

### 3) Procedural
“How do I…”, debugging, setup, deployment, making code work.

---

## Difficulty Levels (MANDATORY)

For each query type, generate exactly **three** queries:

### A) Grounded (Correct)
Well-formed and accurate.

### B) Medium (Slightly Imprecise)
Mostly right but vague/mixed concepts, typical learner phrasing.

### C) Hard (Confused / Failure-Driven)
Mis-specified or symptom-based but still answerable from the transcript.

Hard-mode should include patterns like:
- symptom first (“loss is NaN”, “keeps throwing import error”, “runs forever”)
- wrong framing (“is this even an LLM?”, “does LoRA retrain all params?”)
- mixed terminology (“transfer learning vs LoRA”, “PEFT adds layers?”)

---

## Realism Rules (ENFORCED)

### Coverage constraints (comment-shaped)
Across the 9 queries, include at least:
- **2 debugging/error queries** (concrete symptoms or error-message-ish phrasing)
- **1 deployment/packaging query** (API/UI/serving/export)
- **1 hardware/compute constraint query** (GPU memory, local training, runtime)
- **1 conceptual “what even is this?” query** (LLM definition / confusion)
- **1 tradeoff query** (RAG vs fine-tuning or combining approaches)

### Non-question behavior
- At least **4/9** queries must be **fragments** (not full questions).

### Lexical mismatch (to stress BM25)
- At least **3/9** queries must avoid the transcript’s “main keywords” and instead use
  synonyms or lay phrasing (still answerable from the transcript).

### Typos (controlled)
- Exactly **2/9** queries may include **one** realistic typo (medium or hard only).

### Diversity / no near-duplicates
- No two queries should be paraphrases of each other.
- Each query should target a different “reason someone would search” (setup, definition, math intuition, saving/loading, deployment, tradeoff, metrics, etc.)

---

## Answerability Constraint (STRICT)

Every query must be answerable from the transcript.

If a comment asks something not covered, you may borrow its *style*,
but you must reframe the query so it is answerable using transcript content.

---

## Output Requirements (STRICT)

### Exact Count
- **Exactly 9 queries total**
- Distribution:
  - 3 exact / factual
  - 3 conceptual
  - 3 procedural
- For each type:
  - 1 grounded
  - 1 medium
  - 1 hard

### Output Structure (Pydantic-Friendly)

Return a JSON list of 9 objects.

For each query, output **exactly these fields**:

1) **Query:** natural search-like text  
2) **Type:** exact / conceptual / procedural  
3) **Difficulty:** grounded / medium / hard  
4) **Grounding:** 
   - A short section label you create (e.g., "LoRA intuition", "PEFT definition", "Trainer setup", "Device selection", "Saving/loading")
   - Plus a **≤12-word evidence quote** copied verbatim from the transcript

Format for Grounding:
`"<section label> — <evidence quote>"`

Notes:
- Evidence quote must be ≤12 words and copied verbatim.
- Grounding must make it easy to verify the query is answerable.
