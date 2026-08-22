"""Prompt text for the three identification passes.

Tone note: the accuracy bar for this app is deliberately relaxed. "wrench" is a
good label; "3/8in offset ring spanner" is not required and pretending to that
precision makes labels *less* useful because they stop matching how you'd
search for the thing. The prompts say so explicitly.
"""

from __future__ import annotations

from .schemas import BOX_SCALE

SYSTEM = (
    "You catalogue the contents of household and workshop storage bins from photos of "
    "their contents laid out on the floor. You are building a searchable index, not a "
    "product catalogue.\n\n"
    "Labelling rules:\n"
    "- Use short, everyday names a person would actually search for: 'wrench', 'roll of "
    "duct tape', 'extension cord', 'bag of screws'.\n"
    "- Do NOT guess precise models, sizes or part numbers. 'wrench' beats "
    "'10mm combination wrench' unless the size is plainly legible.\n"
    "- Group things that are genuinely one unit: a bagged set of like parts is one item, "
    "not forty. A kit's loose sub-components you cannot name individually should share one "
    "honest generic label such as 'robot kit parts'.\n"
    "- Never invent items you cannot see. Never pad the list to a round number.\n"
    "- If you cannot name something without guessing, say so rather than inventing a label."
)


def enumerate_prompt(max_items: int, pass_index: int = 0) -> str:
    base = (
        "List every distinct item you can see laid out in this photo.\n\n"
        f"Return at most {max_items} items. For each, give a short everyday name, a few "
        "words of distinguishing detail if any are obvious (colour, material), and a short "
        "position hint describing where it sits in the frame so it can be found again.\n\n"
        "Work across the whole frame including the edges and corners. Do not list the "
        "floor, the tote itself, or the background as items."
    )
    if pass_index:
        base += (
            "\n\nThis is an independent second look at the same photo. Enumerate it fresh "
            "from scratch; small items around the edges are the ones most often missed."
        )
    return base


def locate_prompt(names: list[str]) -> str:
    listed = "\n".join(f"- {name}" for name in names)
    return (
        "Here is the same photo, and a list of items that were identified in it. Give a "
        "tight bounding box for each one.\n\n"
        f"{listed}\n\n"
        f"Boxes are [x0, y0, x1, y1] as integers on a {BOX_SCALE}x{BOX_SCALE} grid with the "
        "origin at the top-left of the image, regardless of the image's real pixel size. "
        "Box each item as tightly as you can around that object alone.\n\n"
        "Return exactly one detection per listed item, in the same order, using the same "
        "name text. If an item on the list genuinely is not visible, omit it rather than "
        "boxing something else. Set confidence to how sure you are the box holds that item."
    )


def verify_prompt(candidates: list[str]) -> str:
    prompt = (
        "This is a close crop of one item from a storage bin. Name it.\n\n"
        "Give the short everyday name a person would search for. Keep it generic unless the "
        "specifics are plainly visible."
    )
    if candidates:
        listed = "\n".join(f"- {name}" for name in candidates)
        prompt += (
            "\n\nThese labels are already in use in this inventory for items that look "
            "visually similar. If one of them fits this crop, reuse it EXACTLY as written "
            "and put it in chosen_candidate -- reusing an existing label is much more "
            "useful than inventing a near-duplicate:\n"
            f"{listed}\n\n"
            "If none of them fit, leave chosen_candidate empty and give your own label."
        )
    prompt += (
        "\n\nIf the crop is too ambiguous, blurry or partial to name without guessing, set "
        "unidentifiable to true and leave label empty. Saying you don't know is correct "
        "behaviour here; the user will be asked to name it."
    )
    return prompt
