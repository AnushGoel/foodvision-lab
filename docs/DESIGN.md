# Design system

The app is a lab report you can interact with: evidence first, plain language, every number with its uncertainty.

## Colour tokens
| Token | Hex | Meaning |
| --- | --- | --- |
| Ink | `#1B2430` | text |
| Muted | `#5B6573` | notes and secondary text |
| B2 / B3 / B4 | `#9DB7DA` / `#4F75B3` / `#1F3A68` | model size as an ordinal light-to-dark ramp (darker = bigger network) |
| Pizza / Steak / Sushi | `#C8402F` / `#7A4E36` / `#2F6B5B` | tomato, char and nori: classes always use their food colour |
| Saffron | `#E3A21A` | the selected model, and only the selected model |

## Typography
IBM Plex Serif for page titles and the overview headline (a research-publication voice); IBM Plex Sans for prose;
Streamlit's default sans-serif for widgets. Line length stays under about 80 characters.

## Visualisation rules
* Every accuracy is shown with an interval (Wilson or bootstrap); point estimates alone are never the evidence.
* Model colour encodes size; class colour encodes food; saffron is reserved for the selection.
* Heat maps (Grad-CAM) are always shown next to the plain model view, and captioned as correlational.
* Charts answer one question each, stated in the section heading.

## Copy
Sentence case, active voice, user language ("Try the model", not "Inference endpoint"). Empty states say what to run;
errors say what went wrong and how to fix it.
