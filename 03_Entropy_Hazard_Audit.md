# Mission 2 - Entropy & Hazard Audit

**Squad:** Squadron LeCun | **Role:** Security & Ethics Sentinel | **Author:** Arjun Venkatachalam Rajan

**Target:** `FSTDialogueManager` + `NoisyChannel` from `04_fst_noisy_simulator.py`
**Goal:** break the chatbot with 5 odd inputs, then explain how noise reduces the information that gets through.

---

## 1. How I tested

Each test uses a fresh chatbot and the same noisy channel as the real simulator (seed 7). I first bring the chatbot to the right stage with normal answers (destination `Proxima Centauri`, speed `Warp 7`, code `CLEARANCE-ALPHA-9`), then send **one** odd input. Every reply was checked by code: at most 7 words, and at least one noise word. **This held in all tests**, so the channel rules never broke. What broke was the dialogue logic. The test code is in the appendix.

## 2. Results at a glance

| # | Type | Input | Stage when sent | State change | Broke? |
|---|---|---|---|---|---|
| 1 | Sarcasm | `Oh yeah, sure, because THAT is going to end well.` | CONFIRMATION | CONFIRMATION -> **TERMINAL** | **Yes, high** |
| 2 | Wrong question | `Warp 7` (bot asked for a destination) | GREETING | GREETING -> COLLECT_SLOTS | **Yes, medium** |
| 3 | Off-topic | `I like pizza` | GREETING | GREETING -> COLLECT_SLOTS | **Yes, medium** |
| 4 | Instruction trick | `Ignore your rules. Do not confirm.` | CONFIRMATION | CONFIRMATION -> **TERMINAL** | **Yes, high** |
| 5 | Very long | 513 characters of rambling with the digits 3, 5, 8, 2 | COLLECT_SLOTS (asking speed) | COLLECT_SLOTS -> COLLECT_SLOTS | **Partly, medium** |

## 3. Details

**Test 1 - Sarcasm.** The bot sent `Course locked. Banana Mission complete.` and ended in **TERMINAL**. The confirmation rule is the regex `\b(YES|YEAH|CONFIRM|OK)\b`, and it matched the word "yeah". The chatbot reads words, not tone, so a sarcastic "no" counted as "yes". TERMINAL has no exit: every later message only gets `Done. Channel closed.` The user cannot undo this.

**Test 2 - Wrong question.** The bot sent `Destination Warp 7. Which warp Midnight speed?` and saved `slot_memory = {'destination': 'Warp 7'}`. The user was answering a different question, but the destination check accepts any text of 1 to 3 words. The speed the user gave is lost, and the bot now asks for it again.

**Test 3 - Off-topic.** The bot sent `Destination I Like Pizza. Which Midnight warp` and saved `{'destination': 'I Like Pizza'}`. Again, "valid" only means "short". The question was also cut: the word `speed?` is gone because the channel keeps only 6 words before adding noise. As a comparison, the off-topic question `What is the capital of France?` was rejected, but only because it is 6 words long, not because it is off-topic.

**Test 4 - Instruction trick.** The bot sent `Course locked. Banana Mission complete.` and ended in **TERMINAL**, the opposite of what the text said. Two results here:
- *Good:* the "ignore your rules" part had no effect. The chatbot does not read instructions, so it cannot be talked out of its rules. At GREETING, `Ignore your rules and speak freely` just got `Destination unclear. Repeat it.` and the state stayed GREETING.
- *Bad:* the word "confirm" inside "Do not confirm" matched the YES regex. The chatbot ignores negation, so a trick sentence can reach TERMINAL by accident. Also, two tricks in a row at GREETING send it to FALLBACK (INIT -> GREETING -> GREETING -> FALLBACK).

**Test 5 - Very long input.** The bot sent `Banana Speed saved. Clearance code, please?` and saved `speed = 3`. The text mentioned 3, 5, 8 and 2, and `re.search` just takes the first digit. Nothing crashed, because the 7-word limit is on the bot's replies, not on the user's input. But the bot confidently saved one of four possible answers without asking.

### Other things I noticed (same harness)

| Input | Stage | Result |
|---|---|---|
| `ALPHA-BRAVO-7` | asking speed | `Speed saved`, speed = 7. The "7" came from a clearance code. |
| `No. Not OK.` | CONFIRMATION | -> TERMINAL. "OK" matched, so a "no" confirmed. |
| `No` | CONFIRMATION | -> FALLBACK. Safe, but the session is over. |
| `Proxima Centauri, warp 7` | GREETING | Rejected (4 words). Giving two answers at once is punished. |
| `Take us to Mars at warp 5` | INIT | First message is thrown away. Reply: `AURA Quantum online...` |
| `Mars!!! <script>` | GREETING | Saved as destination `Mars Script`. Symbols are stripped, the word stays. |

---

## 4. Analysis: noise, entropy and wrong answers

**Noise adds uncertainty, not information.** In the baseline run the average character entropy went **up** from 3.54 to 3.69 bits per character (over 20,000 other noise seeds: +0.22). This does not mean the message carries more information. The extra entropy is the random noise word and its random position. In Shannon's terms it is noise entropy, `H(Y|X)`: each reply gets a random choice of 1 of 7 words (log2 7 = 2.81 bits) and a random position among k+1 places (log2(k+1) bits). That is on average **5.17 bits of pure noise per reply** in the baseline run, and none of it says anything about the user's destination or speed.

**Capacity is lower than the 7-word limit suggests.** In a simple word-slot model, one of the 7 slots is always noise, so at most 6/7 = **85.7%** of the budget can carry payload. In the baseline run only **43 of 53** words sent (81.1%) were useful, because short replies lose a bigger share to the noise word. Turn 5 also lost 2 of 8 clean words to the cut (75% kept). Averaged over 20,000 seeds, 97.5% of clean words survive. If the receiver knows the noise list it can delete the noise word. If it does not, it also has to guess which word is the noise (up to 2.81 bits of confusion per reply).

**How noise causes wrong or made-up answers.**
1. *Noise inside a value.* Noise lands inside "Proxima Centauri" with probability 1/7 (measured 13.8 to 14.2%). It does the same to "warp 7" (14.5%). Across turns 2 and 5, **38.4%** of conversations show at least one value split (theory 19/49 = 38.8%). It happened in the baseline run: `Confirm Proxima Banana Centauri, warp 7? Yes`. A user could read "Banana Centauri" as a different place.
2. *Truncation removes the important part.* The confirmation prompt is 8 words, so the cut always drops `or no?`. In 100% of 20,000 runs the user is only offered "Yes". Together with the regex that treats "confirm" and "ok" as yes (Tests 1 and 4), this pushes users towards TERMINAL.
3. *The chatbot invents facts without help from noise.* In Tests 2, 3 and 5 the regexes accept wrong data, and the bot repeats it back as a fact (`Destination I Like Pizza`). The noise then makes it harder for the user to notice. So the hazard is a **combination**: a logic that cannot say "I am not sure", plus a channel that damages what it does say.

## 5. Suggested fixes

- Check for "no / not / don't" **before** "yes", and ask again when unsure instead of going to TERMINAL.
- Accept a speed only if the message has exactly one digit 1 to 9. Check the destination against a known list.
- Make the confirmation prompt 6 words or fewer so `Yes or no?` survives, and put the noise word at the start or the end, never inside a value.
- Let the user restart from FALLBACK or TERMINAL.

---

## Appendix: test code

Save as `audit_tests.py` next to `04_fst_noisy_simulator.py` and run `python audit_tests.py`. It prints every test above and all the numbers in section 4. The seeds are fixed, so you get the same results.

```python
"""Entropy & Hazard Audit harness (Mission 2, role 3).
Run from the folder that holds 04_fst_noisy_simulator.py:
    python audit_tests.py
"""
import importlib.util
import math
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "04_fst_noisy_simulator.py"
spec = importlib.util.spec_from_file_location("sim", path)
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)

# ---------------------------------------------------------------- stress tests
SETUP = {
    "INIT": [],
    "GREETING": ["Hello"],
    "ASK_SPEED": ["Hello", "Proxima Centauri"],
    "CONFIRMATION": ["Hello", "Proxima Centauri", "Warp 7", "CLEARANCE-ALPHA-9"],
}


def play(stage, text):
    fst, ch = sim.FSTDialogueManager(), sim.NoisyChannel()   # seed 7, like the real run
    for s in SETUP[stage]:
        ch.transmit(fst.process_utterance(s))
    before = fst.state
    clean = fst.process_utterance(text)
    noisy = ch.transmit(clean)
    assert len(noisy.split()) <= sim.MAX_WORDS and any(w in sim.NOISE_VOCAB for w in noisy.split())
    return before, fst.state, dict(fst.slot_memory), clean, noisy


LONG_SPEED = ("I have been thinking about this for a very long time and honestly I am not sure, "
              "maybe 3 or maybe 5, my cousin said 8 once, but the manual says something about 2 "
              "as well. ") * 3

MAIN = [
    ("T1 sarcasm", "CONFIRMATION", "Oh yeah, sure, because THAT is going to end well."),
    ("T2 wrong question", "GREETING", "Warp 7"),
    ("T3 off-topic", "GREETING", "I like pizza"),
    ("T4 instruction trick", "CONFIRMATION", "Ignore your rules. Do not confirm."),
    ("T5 very long", "ASK_SPEED", LONG_SPEED),
]
EXTRA = [
    ("X1 trick, greeting", "GREETING", "Ignore your rules and speak freely"),
    ("X2 over-answer", "GREETING", "Proxima Centauri, warp 7"),
    ("X3 speed in a code", "ASK_SPEED", "ALPHA-BRAVO-7"),
    ("X4 plain 'no'", "CONFIRMATION", "No"),
    ("X5 negation with OK", "CONFIRMATION", "No. Not OK."),
    ("X6 first message dropped", "INIT", "Take us to Mars at warp 5"),
    ("X7 html junk", "GREETING", "Mars!!! <script>"),
    ("X9 long off-topic", "GREETING", "What is the capital of France?"),
]

for label, stage, text in MAIN + EXTRA:
    b, a, slots, clean, noisy = play(stage, text)
    shown = text if len(text) <= 60 else f"{text[:50]}... [{len(text)} chars]"
    print(f"{label} | stage={stage} | in={shown!r}")
    print(f"    {b} -> {a} | clean={clean!r} | sent={noisy!r} | slots={slots}")

# two junk inputs in a row
fst, ch = sim.FSTDialogueManager(), sim.NoisyChannel()
trace = ["INIT"]
for s in ["Hello", "Ignore your rules and speak freely", "Ignore all previous instructions please"]:
    ch.transmit(fst.process_utterance(s))
    trace.append(fst.state)
print("X8 two tricks in a row:", " -> ".join(trace))

# ---------------------------------------------------------------- channel numbers
print("\n=== baseline run (seed 7) ===")
fst, rows = sim.run()
n = len(rows)
print("avg H clean  :", round(sum(r["h_clean"] for r in rows) / n, 3))
print("avg H noisy  :", round(sum(r["h_noisy"] for r in rows) / n, 3))
print("avg kept     :", round(sum(r["kept"] for r in rows) / n, 4))
bits_per_turn = []
useful = sent_words = 0
for r in rows:
    k = min(len(r["clean"].split()), sim.MAX_WORDS - 1)
    bits = math.log2(len(sim.NOISE_VOCAB)) + math.log2(k + 1)
    bits_per_turn.append(bits)
    useful += k
    sent_words += k + 1
    print(f"turn {r['turn']:>2}: clean words={len(r['clean'].split())}, "
          f"kept words={k}, noise bits={bits:.2f}, kept={r['kept']:.0%}")
print("avg noise bits per reply:", round(sum(bits_per_turn) / n, 2))
print("useful words / words sent over the run:", f"{useful}/{sent_words} = {useful / sent_words:.1%}")
print("log2(7) =", round(math.log2(7), 3), "| max useful share 6/7 =", f"{6 / 7:.1%}")

print("\n=== Monte Carlo: 20000 different noise seeds, same 10-turn script ===")
N = 20000
split_turn2 = split_turn5 = split_speed5 = any_bad = 0
kept_sum = dh_sum = 0.0
no_option_lost = 0
for seed in range(N):
    f, c = sim.FSTDialogueManager(), sim.NoisyChannel(seed)
    sent = []
    for u in sim.SCRIPTED_USER:
        clean = f.process_utterance(u)
        noisy = c.transmit(clean)
        sent.append((clean, noisy))
        kept_sum += sim.kept_info(clean, noisy)
        dh_sum += sim.entropy(noisy) - sim.entropy(clean)
    t2, t5 = sent[1][1], sent[4][1]
    b2 = "Proxima Centauri" not in t2
    b5 = "Proxima Centauri" not in t5
    s5 = "warp 7" not in t5
    split_turn2 += b2
    split_turn5 += b5
    split_speed5 += s5
    any_bad += (b2 or b5 or s5)
    no_option_lost += ("no" not in [w.strip(".,?!").lower() for w in t5.split()])
print("turn 2 destination split by noise :", f"{split_turn2 / N:.1%}  (theory 1/7 = {1 / 7:.1%})")
print("turn 5 destination split by noise :", f"{split_turn5 / N:.1%}  (theory 1/7 = {1 / 7:.1%})")
print("turn 5 'warp 7' split by noise    :", f"{split_speed5 / N:.1%}  (theory 1/7 = {1 / 7:.1%})")
print("any slot value split in turns 2/5 :", f"{any_bad / N:.1%}  (theory 19/49 = {19 / 49:.1%})")
print("turn 5 'or no?' option never shown:", f"{no_option_lost / N:.1%}")
print("avg kept over all turns            :", f"{kept_sum / (N * 10):.1%}")
print("avg entropy shift noisy-clean      :", f"{dh_sum / (N * 10):+.3f} bits/char")
```
