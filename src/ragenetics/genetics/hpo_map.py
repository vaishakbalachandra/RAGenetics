from typing import List

# Tiny demo lexicon (extend with your KB)
PHRASE_TO_HPO = {
    "short stature": "HP:0004322",
    "seizures": "HP:0001250",
    "recurrent infections": "HP:0002719",
    "diarrhea": "HP:0002027",
}


def extract_hpo_phrases(text: str) -> List[str]:
    """
    Extracts matching HPO codes from input text.

    Args:
        text (str): The input clinical text.

    Returns:
        List[str]: List of matched HPO codes.
    """
    text = text.lower()
    out = []
    for phrase, hpo_code in PHRASE_TO_HPO.items():
        if phrase in text:
            out.append(hpo_code)
    return out
