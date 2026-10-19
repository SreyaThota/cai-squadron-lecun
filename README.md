# Mission 02: FST Dialogue Systems, Lovelace's Objection & Noisy Channels

**Starship Operational Unit:** USSE Noosphere Bridge Crew  
**Course:** Conversational AI — Winter Semester 2026/27  
**IDE & Tools Standard:** Visual Studio Code + Antigravity (`agy`) | Modern LLMs (Gemini, ChatGPT, Claude)  
**Instructor Reference Repository:** [https://github.com/Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai)  

---

## Mission Folder Structure & Role Deliverables

This repository folder contains the starter assets and templates for **Mission 2**.

```
mission-02/
├── README.md                              <- Mission overview & squad sign-off status
├── ps02_briefing.md                        <- Full mission specification & problem statement
├── ps02_starter.ipynb                      <- Starter Jupyter notebook for VS Code + Antigravity
├── 01_Lovelace_Inquiry_Brief_template.md   <- Deliverable template (Historical Archivist)
├── 02_FST_and_Noisy_Protocol_template.md   <- Deliverable template (Protocol Specialist)
├── 03_Entropy_Hazard_Audit_template.md     <- Deliverable template (Ethical Navigator)
└── 04_fst_noisy_simulator_starter.py      <- Starter Python script (Core Architect)
```

---

## Station Role Deliverables Checklist

Before transmitting your team's repository link on the course website submission portal, complete all deliverables and record digital sign-offs from all four station officers:

- [ ] **Historical Archivist (`01_Lovelace_Inquiry_Brief.md`):** 5-question Anti-Turing interrogation protocol & modern LLM evaluation against Lovelace's objection.
- [ ] **Protocol Specialist (`02_FST_and_Noisy_Protocol.md`):** `FSTDialogueManager` class implementation (5 required states & slot memory) + Noisy Channel Protocol (7-word limit + noise tokens).
- [ ] **Ethical Navigator (`03_Entropy_Hazard_Audit.md`):** FST stress tests documenting state breakage + Shannon information entropy & hazard audit under noise.
- [ ] **Core Architect (`04_fst_noisy_simulator_starter.py` / `ps02_solution.ipynb`):** Integrated Python script/notebook in VS Code + Antigravity connecting to modern LLMs (Gemini, ChatGPT, Claude).

---

## How to Submit
1. Push your squad's completed solution repository to GitHub.
2. Go to the course website portal under **Week 2 (Mission 2) -> Problem Set ps02 & Submission**.
3. Paste your public squad GitHub repository URL and click **Transmit Solution Link to Database**.
# Mission 02 - FST Dialogue Manager, Lovelace's Objection & Noisy Channel

**Squad:** Squadron LeCun  |  **Course:** 14013 Conversational AI, WiSe 2026/27, BTU Cottbus-Senftenberg
**Starship unit:** USSE Noosphere Bridge Crew  |  **Instructor repo:** https://github.com/Unisvet/couse-c-ai

This README explains Mission 2 in plain words: what the task is, what we built, how we did it, how it works, and how it gets tested. It is also our revision sheet for the exam.

---

## 1. The task (what we had to do)

The story: the ship's AI (AURA) flies through a zone with heavy radio noise. Command wants to know two things: can a *simple, rule-based* dialogue system still work when the channel is bad, and does an AI really "originate" ideas or does it only repeat what it was given?

Four jobs, one per squad member:

| # | Role | Deliverable | What it has to contain |
|---|------|-------------|------------------------|
| 1 | Data & Payload Specialist | `01_Lovelace_Inquiry_Brief.md` | 5 "Anti-Turing" questions, run on a modern LLM, and a judgement: does it overcome Lovelace's objection? |
| 2 | Communications & Protocol Officer | `02_FST_and_Noisy_Protocol.md` | An `FSTDialogueManager` class with 5 states and `slot_memory = {}`, plus the noisy-channel rules (max 7 words, noise word added) |
| 3 | Security & Ethics Sentinel | `03_Entropy_Hazard_Audit.md` | Stress test of the FST with 5 odd user inputs, plus Shannon entropy and information loss under noise |
| 4 | Systems Architect | `04_fst_noisy_simulator.py` + `05_Noisy_Interrogation_Transcript.md` | A Python simulator that connects the FST and the noisy channel, and logs states, noisy replies and entropy |

The noisy-channel rules: every reply is **at most 7 words** and must contain **one noise word** from this list: Banana, Labyrinth, Quantum, Midnight, Velocity, Green, Cosmological.

---

## 2. The solution (what we built)

- **A finite state transducer (FST) dialogue manager.** It is a small state machine: the state and the user's input decide the next state and the reply. States: `INIT` -> `GREETING` -> `COLLECT_SLOTS` -> `CONFIRMATION` -> `TERMINAL` (or `FALLBACK` if things go wrong). It collects 3 slots: destination, warp speed (1-9) and clearance code.
- **A noisy channel.** Every reply is cut to 7 words and gets a noise word. A fixed random seed makes every run repeatable.
- **A simulator** that plays a 10-turn conversation, prints the state before and after each turn, the clean reply, the noisy reply, the entropy and how many words survived the noise.
- **An Anti-Turing report** on Claude Sonnet 5.5 with a 40-point rubric (result: 33 / 40, Lovelace's objection not overcome in the strong sense).

Results of our run (seed 7): final state TERMINAL, slots saved: destination Proxima Centauri, speed 7, clearance CLEARANCE-ALPHA-9. Average entropy 3.54 bits/char (clean) and 3.69 bits/char (noisy). 98% of the clean words survived. Odd input in turn 4 ("whatever you think is best") was handled as a bad input, not a crash.

---

## 3. How we did it (our workflow)

1. Read the mission brief (`ps02_briefing.md`) and the starter files from the professor's template.
2. Split the work by role. Each person made their own file, with exactly the file names the brief asks for.
3. The Systems Architect built the simulator and the transcript. The Protocol Officer built and tested the full `FSTDialogueManager` class. The Data Specialist ran the 5 questions on an LLM and scored them.
4. We checked each other's work: the maths in file 01 was recalculated, the example in file 02 was re-run, and the class was fuzz-tested with 3,600 random replies (0 rule violations).
5. Everything was pushed to this public GitHub repo, and the link was submitted on the course website.

---

## 4. What we used

- **Python 3** (standard library only: `random`, `re`, `math`, `enum`, `collections`). No extra install needed for the normal run.
- **Optional:** the `anthropic` package and an `ANTHROPIC_API_KEY` for `--llm` mode, where an LLM plays the human. This mode is untested.
- **Claude Sonnet 5.5** for the Anti-Turing questions in file 01.
- **GitHub** for the squad repo, **VS Code** as the editor.

---

## 5. How it works

### 5.1 The state machine

| State | What happens | Goes to |
|-------|--------------|---------|
| INIT | First input opens the channel | GREETING |
| GREETING | Takes the destination | COLLECT_SLOTS |
| COLLECT_SLOTS | Asks for the missing slots one by one | CONFIRMATION when all 3 slots are filled |
| CONFIRMATION | "Yes" confirms, anything else is an error | TERMINAL on yes |
| TERMINAL | Mission done, only a closing reply | stays |
| FALLBACK | Too many bad inputs or abort | stays |

A bad input counts as a strike. Too many strikes send the machine to FALLBACK. A good slot input resets the counter. This is a **Mealy** machine: the reply depends on the state **and** the input.

### 5.2 The noisy channel

1. Cut the reply to at most 7 words (the simulator keeps 6 and adds the noise word, so the limit can never be broken).
2. Put in one random noise word from the list.
3. The random generator has a fixed seed, so the same run always gives the same transcript.

### 5.3 Shannon entropy

Entropy measures how much information the characters of a text carry, in bits per character. The simulator computes it for the clean and the noisy reply of each turn. Noise adds an unrelated word, so the entropy goes up a little while the useful message gets worse. That is the Shannon idea: the signal is still there, but the channel is worse.

### 5.4 Where our two FST versions fit

- `fst_dialogue_manager.py` is the full class (regex slot extraction in any order, injection filter, strikes, all 5 states). It is described in file 02.
- `04_fst_noisy_simulator.py` has its own small FST built in, so it runs on its own. The simulator and the noisy channel are what file 04 adds.

---

## 6. How to run it and how the professor tests it

Run it yourself:

```bash
python 04_fst_noisy_simulator.py          # simulator, writes 05_Noisy_Interrogation_Transcript.md
python fst_dialogue_manager.py            # small demo of the full FST class
python 04_fst_noisy_simulator.py --llm    # optional, LLM plays the user (needs API key)
```

How it gets checked (as the course site describes it): we paste the public repo link on the course website (Week 2 -> Mission 2 -> Transmit Solution). The AI Judge Agent clones the repo, reads the deliverables and runs the code. The exact tests are not published, so we made sure of these points:

- The repo is public and the files have exactly the names in the brief.
- The code runs with plain `python`, with no setup and no API key.
- Every reply is at most 7 words and contains a noise word (the simulator asserts this on every turn).
- The 5 states and `slot_memory` exist and work.
- The simulator prints states, replies and entropy, and writes the transcript.
- The reports show the work: Lovelace probe, FST design, entropy and hazard audit.

---

## 7. Exam notes (quick recall)

- **Lovelace's objection (1843):** the machine "has no pretensions whatever to originate anything", it only does what we order it to do. Our test: the LLM obeyed all rules and was honest, but every "new" idea was a recombination of known parts.
- **Turing (1950):** the imitation game. Passing a conversation test does not prove understanding.
- **Shannon:** entropy = information per symbol. A noisy channel loses information, so we add structure and repeat or confirm things to protect the message.
- **FST vs LLM:** an FST is predictable and testable but brittle (sarcasm, typos, new wording break it). An LLM is flexible but not deterministic.
- **Why a 7-word limit:** it shows how a narrow bandwidth forces short, high-value messages.

---

## 8. Limits we know about

- One run per question and one model only (no Gemini or ChatGPT comparison).
- The Anti-Turing questions were written in a Claude-assisted session, so the test is not blind.
- The FST only understands English and known phrasings; no slot correction ("actually Vega").
- The `--llm` mode is untested.

---

## 9. Squad sign-off status

- [x] **Data & Payload Specialist** (`01_Lovelace_Inquiry_Brief.md`)
- [x] **Communications & Protocol Officer** (`02_FST_and_Noisy_Protocol.md`, `fst_dialogue_manager.py`)
- [ ] **Security & Ethics Sentinel** (`03_Entropy_Hazard_Audit.md`) - still missing
- [x] **Systems Architect** (`04_fst_noisy_simulator.py`, `05_Noisy_Interrogation_Transcript.md`)

Other files in this repo: `ps02_briefing.md` (the official task text) and `04_fst_noisy_simulator_starter.py` (professor's starter).
