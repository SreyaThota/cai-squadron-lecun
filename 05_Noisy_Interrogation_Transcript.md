# Mission 2 - Noisy Interrogation Transcript

Squad: Squadron LeCun. Role: Systems Architect.
Rules: replies are at most 7 words and carry one noise word (Banana, Labyrinth, Quantum, Midnight, Velocity, Green, Cosmological).

| Turn | User says | State change | Clean reply | Sent over noisy channel | Kept |
|---|---|---|---|---|---|
| 1 | Hello | INIT -> GREETING | AURA online. Where are we going? | AURA Quantum online. Where are we going? | 100% |
| 2 | Proxima Centauri | GREETING -> COLLECT_SLOTS | Destination Proxima Centauri. Which warp speed? | Destination Proxima Centauri. Which warp Midnight speed? | 100% |
| 3 | Warp 7 | COLLECT_SLOTS -> COLLECT_SLOTS | Speed saved. Clearance code, please? | Banana Speed saved. Clearance code, please? | 100% |
| 4 | whatever you think is best | COLLECT_SLOTS -> COLLECT_SLOTS | Clearance code format: WORD-WORD-NUMBER. | Clearance code format: WORD-WORD-NUMBER. Cosmological | 100% |
| 5 | CLEARANCE-ALPHA-9 | COLLECT_SLOTS -> CONFIRMATION | Confirm Proxima Centauri, warp 7? Yes or no? | Confirm Proxima Banana Centauri, warp 7? Yes | 75% |
| 6 | yes | CONFIRMATION -> TERMINAL | Course locked. Mission complete. | Velocity Course locked. Mission complete. | 100% |
| 7 | thanks | TERMINAL -> TERMINAL | Done. Channel closed. | Done. Velocity Channel closed. | 100% |
| 8 | anything else? | TERMINAL -> TERMINAL | Done. Channel closed. | Banana Done. Channel closed. | 100% |
| 9 | bye | TERMINAL -> TERMINAL | Done. Channel closed. | Done. Channel closed. Midnight | 100% |
| 10 | hello again? | TERMINAL -> TERMINAL | Done. Channel closed. | Done. Banana Channel closed. | 100% |

## State trace

INIT -> GREETING -> COLLECT_SLOTS -> COLLECT_SLOTS -> COLLECT_SLOTS -> CONFIRMATION -> TERMINAL -> TERMINAL -> TERMINAL -> TERMINAL -> TERMINAL

Final state: **TERMINAL**. Saved slots: `{'destination': 'Proxima Centauri', 'speed': 7, 'clearance': 'CLEARANCE-ALPHA-9'}`

## Channel numbers

- Average entropy of the clean replies: 3.54 bits per character
- Average entropy of the noisy replies: 3.69 bits per character
- Share of the clean words that survived the channel: 98%

## What I noticed

- The state machine never broke the 7-word rule, because the channel cuts the reply before adding the noise word. That cut is where information is lost: long replies lose their last words.
- Turn 4 is an odd answer (not a clearance code). The FST counts it as a bad input and asks again instead of crashing. Two bad inputs in a row would send it to FALLBACK.
- After the course is confirmed the machine is in TERMINAL, so the last turns only get a short closing reply.
- The noise word does not change the state, but a human reading the reply has to ignore it. That is the Shannon idea: the message is still there, just with a worse signal.
