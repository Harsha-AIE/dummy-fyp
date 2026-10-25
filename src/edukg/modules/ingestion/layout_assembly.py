from __future__ import annotations


def assemble_layout_text(blocks: list[dict]) -> str:
    """Joins OCR blocks (already reading-order sorted by the OCRProvider) into text,
    inserting a paragraph break whenever the vertical gap between blocks is large
    enough to suggest a new line/paragraph rather than same-line continuation."""
    if not blocks:
        return ""

    lines: list[str] = []
    previous_bottom: float | None = None

    for block in blocks:
        bbox = block["bbox"]
        top = min(p[1] for p in bbox)
        bottom = max(p[1] for p in bbox)
        text = block["text"].strip()
        if not text:
            continue

        if previous_bottom is not None and (top - previous_bottom) > (bottom - top) * 0.5:
            lines.append("")  # paragraph break
        lines.append(text)
        previous_bottom = bottom

    return "\n".join(lines)
