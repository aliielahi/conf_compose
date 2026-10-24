"""Final answers in \\boxed{...}: extraction, normalisation and equality."""


def boxed(text):
    """Content of the last \\boxed{...} in text, or None."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth, i = 1, start + len("\\boxed{")
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i:j]
    return None


def normalise(answer):
    """Answer string with formatting that does not change its value removed."""
    if answer is None:
        return None
    out = answer.strip().strip("$").rstrip(".")
    for old, new in (("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("dfrac", "frac"), ("tfrac", "frac"),
                     (" ", ""), ("^\\circ", ""), ("^{\\circ}", ""), ("\\%", ""), ("%", "")):
        out = out.replace(old, new)
    try:
        return str(round(float(out.replace(",", "")), 6))
    except ValueError:
        return out


def equal(a, b):
    return a is not None and b is not None and normalise(a) == normalise(b)
