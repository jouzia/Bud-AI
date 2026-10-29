**Bud AI x OpenEnv: Multi-Agent Educational Platform & Intelligence Benchmark** 
 
An exploratory multi-agent educational assistant prototype built around a stateful, reward-driven study environment. It is a research/engineering prototype; production readiness has not been established by deployment, security, load, or end-to-end verification.

* **Application Layer:** Bud AI (Next.js / Python Custom Web Interface)
* **Core Engine:** OpenEnv Study Intelligence (Stateful Agent Environment)
* **Status:** Prototype. A public Vercel demo has been linked externally, but this repository review has not yet established end-to-end availability or clean-install reproducibility.

---

## Bud AI Layer: Glassmorphism Frontend Aesthetic
Bud AI provides a high-contrast, ultra-modern developer interface designed around a strict dark-mode layout:
* **Brutalist Architecture:** Deep near-black backgrounds (`#0B0B0F`) to minimize cognitive fatigue.
* **Frosted Interfaces:** UI components built as translucent, frosted glass panels with smooth backdrop blur filters and microscopic white borders.
* **Ambient Signaling:** Subtle neon color underglows that visually pulse to signal different multi-agent execution states (e.g., active routing vs. content streaming).

---

## Core Multi-Agent Logic & Routing Architecture

Bud AI transitions past monolithic single-prompt architectures. It utilizes an internal intent routing core that delegates tasks to specialized, context-aware agent models depending on conversational requirements.


```

[ User Prompt / Input UI ]
│
▼
┌────────────────────────┐
│   Intent Router Core   │ ─── (Parses Scope & Context)
└────────────────────────┘
│
┌─────────┴─────────┐
▼                   ▼
┌─────────────────┐ ┌──────────────────┐
│  Bud Cat Agent  │ │ Academic Teacher │ ... (Other Specialized Agents)
│  (Ambient / UI) │ │ (Data / Coding)  │
└─────────────────┘ └──────────────────┘
│                   │
└─────────┬─────────┘
▼
┌───────────────────────────┐
│ OpenEnv Benchmark Engine  │ ─── (Calculates Step Reward / S)
└───────────────────────────┘

```

---

## OpenEnv Core Engine: The Benchmark Sandbox

Behind the user interface sits **OpenEnv**, a Gym-style environment for AI tutors where the LLM handles execution and the environment enforces underlying strategic reasoning.

### Problem Statement

LLMs are typically evaluated on static benchmarks (MMLU, HumanEval) which measure point-in-time language ability. They fail to evaluate how an agent performs when past decisions compound, constraints tighten, and memory must be leveraged fluidly across steps. OpenEnv fixes this gap.

### Technical Blueprint & System Architecture


```

┌─────────────────────────────────────────────────┐
│             Gradio UI  (app/app.py)              │
│  conversation tree · charts · human-in-the-loop  │
└──────────────────┬──────────────────────────────┘
│  delegates to
┌──────────────────▼──────────────────────────────┐
│         core.benchmark.run_episode()             │
│         orchestration only — zero UI imports     │
└────┬───────────────┬──────────────────┬──────────┘
│               │                  │
┌────▼────┐  ┌───────▼──────┐  ┌───────▼───────┐
│ StudyEnv│  │ MemoryAgent  │  │    Grader     │
│ engine  │  │ memory_agent │  │   grader.py   │
└─────────┘  └──────┬───────┘  └───────────────┘
│
┌────────▼────────┐
│ DualLayerMemory │
│ episodic+semantic│
└─────────────────┘

```

---

## Reward Optimization Formula

The agent environment calculates programmatic feedback based on structured constraints:

$$S = (0.50 \times C) + (0.30 \times D) + (0.20 \times E) - P$$

| Metric Element | Full Component Context | Weight |
| :--- | :--- | :--- |
| **C** | Completeness Metric (Scaled 0.00 to 1.00) | **50%** |
| **D** | Diversity Index — Unique structural action types | **30%** |
| **E** | Efficiency Index — Rate of completeness achieved per step | **20%** |
| **P** | Cumulative Penalty (Triggered by bad/repetitive operations) | *Subtracted* |

> ⚠️ **Penalty Schedule Parameters:** Repeated Actions: `-0.30` | Illegal Conversational State: `-0.50` | Empty Operations: `-0.20`

---

## Deep Feature Ecosystem

* **Dual-Layer Memory:** Built using an episodic ring buffer (retaining 5 historical steps) coupled to a semantic key-value store. Crucial educational facts are systematically indexed via explicit `[FACT: concept | definition]` extraction tags.
* **Complexity Scaling Profiles:** Dynamic system runtime scaling across three functional modes: **ELI5** (Primary education vocabulary focus), **Standard** (University undergraduate level), and **PhD** (Advanced research vocabulary).
* **Human-in-the-Loop Interventions:** Dedicated debugging dashboard allows real-time interactive adjustments mid-session. Corrections are injected directly back into semantic memory at a hardcoded weight of `1.0`.
* **Dynamic Knowledge Graph Rendering:** Generates real-time, vector-mapped network charts via Plotly where every extracted fact morphs dynamically into connected nodes.

---

## Project Navigation Directory


```

openenv/
├── app/app.py                      UI layer only — zero business logic
├── core/
│   ├── config.py                   centralised config (YAML + env vars)
│   ├── telemetry.py                structured logging, session metrics
│   ├── benchmark.py                run_episode(), run_comparative_benchmark()
│   ├── agents/
│   │   ├── base.py                 BaseAgent interface
│   │   ├── memory_agent.py         Gemini + dual-layer memory + ELI5/PhD mode
│   │   ├── gemini_agent.py         plain Gemini agent (comparison baseline)
│   │   └── heuristic_agent.py      deterministic rule-based baseline
│   ├── environment/
│   │   ├── engine.py               StudyEnv — step(), reset(), reward logic
│   │   └── state.py                StudyState (Pydantic v2, frozen/immutable)
│   ├── grading/grader.py           GradeReport, letter grades, breakdown
│   ├── memory/memory.py            DualLayerMemory: episodic ring + semantic KV
│   └── conversation/
│       ├── tree.py                 DecisionTree engine (data-driven, no Gradio)
│       └── openenv_tree.py         5-branch guided conversation node definitions
│   └── visualisation/
│       └── knowledge_graph.py      Plotly network graph from Fact list
├── openenv.yaml                    machine-readable environment specification
├── requirements.txt                pinned production dependencies
└── .env.example                    required environment variables

```

---

## Local Setup (verification required)

```bash
# Clone repository and access directory
git clone [https://github.com/jouzia/Bud-AI.git](https://github.com/jouzia/Bud-AI.git) && cd Bud-AI

# Install environment workspace dependencies 
python -m venv .venv
# Activate the environment, then:
python -m pip install --upgrade pip
pip install -r requirements.txt

# Configure the environment variables required by core/config.py.
# Do not commit API keys or local .env files.
python app/app.py

# If startup fails, check the installed dependencies, model/API configuration,
# and the traceback before treating the local setup as verified.

```

---

## Benchmark Status

This repository does not currently provide a committed, reproducible benchmark-results artifact supporting comparative scores for GPT-4o, Gemini, Llama, and the heuristic baseline. Therefore, no comparative performance numbers are claimed here. Publish scores only with the exact evaluation prompts, model/version identifiers, run counts, scoring code, raw results, and uncertainty analysis.

---

## Next-Gen System Development Roadmap

1. **Interactive Multi-Agent Classrooms:** Active dual-agent loops where one agent drives informational output while a secondary peer agent dynamically injects contextual quizzes.
2. **Dynamic PDF Data Ingestion:** Automated chunking and vector processing of custom textbook and course syllabi files.
3. **Vector Core Migration:** Swapping volatile localized storage arrays for a distributed ChromaDB persistent vector database layer.

---

## Verification Status

- The repository contains application and environment code, but this review has not run it from a clean environment.
- No production-readiness or comparative benchmark claim is made.
- Treat the external demo as unverified until a fresh browser session can complete a representative study interaction and error handling is checked.

License: MIT © Shaik Jouzia Afreen H

```

```
