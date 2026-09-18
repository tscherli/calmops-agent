# Third-party notice: Stretchly break ideas

Rows in `sandbox/measures.csv` whose `source` starts with `stretchly:` are taken from the break
ideas of **Stretchly**, the break-reminder app by Jan Hovancik.

- Repository: https://github.com/hovancik/stretchly
- Read from commit `d33ea65526f7b43c8af7148cc7d254b8c34dcda6` (2026-08-14)
- File: `app/locales/en.json`, keys `miniBreakIdeas` and `longBreakIdeas`
- Licence: BSD 2-Clause, reproduced unchanged in [LICENSE](LICENSE)

The `source` column names the original idea: `stretchly:mini:aad` is `miniBreakIdeas.aad`,
`stretchly:long:abj` is `longBreakIdeas.abj`.

## Changes made

**Used unchanged:** `mini:aaa`, `mini:aap`, `mini:aaq`, `mini:abb`, `mini:aam`, `mini:aax`,
`mini:aay`.

**Wording adapted** (shortened, turned into an instruction, or grammar fixed):

| source | original | used as |
|---|---|---|
| `mini:aad` | Close your eyes and take few deep breaths. | Close your eyes and take a few deep breaths. |
| `mini:aak` | Stand from your chair and stretch. | Stand up from your chair and stretch. |
| `long:abj` | "Overwhelmed?" Try the 5-4-3-2-1 Grounding Technique: Identify 5 things you can see, … | Identify 5 things you can see, 4 you can touch, … |
| `long:abi` | "Imaginative visualization" Close your eyes and imagine yourself in a peaceful and calming place, … | Close your eyes and imagine a peaceful place, … |
| `long:aaj` | "Tech" How about taking a no-tech walk? | Take a short walk without your phone or computer. |
| `long:aab` | "Step away" Do you ever notice how your brain can figure things out by itself? … | Step away from the computer and think about something completely unrelated. |
| `long:abb` | "Lunch outside" Nature is linked to positive emotions … | Take your lunch break outside, ideally somewhere green. |
| `long:aaa` | "Not alone" Do you find it hard to take a break alone? … | Take your break together with a colleague. |

**Added by this project, not part of Stretchly:** the `measure_id`, `type`, `fits_energy`
and `effort` columns. The energy fit is this project's own judgement, not a claim made by Stretchly.

**Deliberately left out:** health or research claims and intense exercise, which don't belong in a
non-clinical tool (e.g. "Pulse", "7 Minute Workout", "Take the stairs", "Meditation", "Make art",
"Metabolism"), plus ergonomics and eye or wrist exercises that address sitting at a screen rather than
work stress.
