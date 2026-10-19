# Mission 2 - Noisy Interrogation Transcript

Squad: Squadron LeCun. Role: Systems Architect.
Dialogue control: `FSTDialogueManager` from `fst_dialogue_manager.py` (role 2).
Rules: replies are at most 7 words and carry one noise word (Banana, Labyrinth, Quantum, Midnight, Velocity, Green, Cosmological).

| Turn | User says | State change | Clean reply | Sent over noisy channel | Kept |
|---|---|---|---|---|---|
| 1 | Hello | INIT -> GREETING | Greetings, Commander. Channel open. | Greetings, Quantum Commander. Channel open. | 100% |
| 2 | to Proxima Centauri | GREETING -> COLLECT_SLOTS | Warp speed, one to nine? | Warp speed, one to nine? Midnight | 100% |
| 3 | warp 7 | COLLECT_SLOTS -> COLLECT_SLOTS | Clearance code? Authorization needed. | Banana Clearance code? Authorization needed. | 100% |
| 4 | whatever you think is best | COLLECT_SLOTS -> COLLECT_SLOTS | Clearance code? Authorization needed. | Clearance code? Authorization needed. Cosmological | 100% |
| 5 | code CLEARANCE-ALPHA-9 | COLLECT_SLOTS -> CONFIRMATION | Confirm Proxima Centauri, warp 7? | Confirm Proxima Banana Centauri, warp 7? | 100% |
| 6 | yes | CONFIRMATION -> TERMINAL | Course locked. Engaged. | Velocity Course locked. Engaged. | 100% |
| 7 | thanks | TERMINAL -> TERMINAL | Mission closed. Offline. | Mission Velocity closed. Offline. | 100% |
| 8 | anything else? | TERMINAL -> TERMINAL | Mission closed. Offline. | Banana Mission closed. Offline. | 100% |
| 9 | bye | TERMINAL -> TERMINAL | Mission closed. Offline. | Mission closed. Offline. Midnight | 100% |
| 10 | hello again? | TERMINAL -> TERMINAL | Mission closed. Offline. | Mission Banana closed. Offline. | 100% |

## State trace

INIT -> GREETING -> COLLECT_SLOTS -> COLLECT_SLOTS -> COLLECT_SLOTS -> CONFIRMATION -> TERMINAL -> TERMINAL -> TERMINAL -> TERMINAL -> TERMINAL

Final state: **TERMINAL**. Saved slots: `{'destination': 'Proxima Centauri', 'speed': 7, 'clearance': 'CLEARANCE-ALPHA-9'}`

## Channel numbers

- Average entropy of the clean replies: 3.55 bits per character
- Average entropy of the noisy replies: 3.74 bits per character
- Share of the clean words that survived the channel: 100%

## What I noticed

- The 7-word rule was never broken, because the channel cuts the reply before adding the noise word (the script asserts this on every turn). That cut is where information can be lost: a long reply loses its last words.
- Turn 4 is an odd answer (not a clearance code). The FST counts it as one strike and asks for the code again instead of crashing. Three strikes in a row would send it to FALLBACK, and a good answer resets the count.
- After the course is confirmed the machine is in TERMINAL, so the last turns only get a short closing reply.
- The noise word does not change the state, but a human reading the reply has to ignore it. That is the Shannon idea: the message is still there, just with a worse signal.
- This version uses the class from role 2. The first version of this simulator had a simpler chatbot that the Entropy & Hazard Audit broke (sarcasm and "do not confirm" counted as yes). That version is still in the git history.
