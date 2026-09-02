import os
import json
import re

from nltk.translate.bleu_score import (
    sentence_bleu,
    corpus_bleu,
    SmoothingFunction,
)


# ============================================================
# Configuration
# ============================================================

REFERENCE_FILE = "Data/VRSBench_EVAL_Cap.json"
GENERATED_FILE = "outputs/generated_captions.txt"
OUTPUT_FILE = "outputs/bleu_scores.txt"

IMAGE_IDS = [
    "P0003_0002.png",
    "P0019_0002.png",
    "P0060_0004.png",
    "P0110_0017.png",
    "P0146_0005.png",
    "P0168_0009.png",
]


# ============================================================
# Tokenization
# ============================================================

def tokenize(text):
    """
    Lowercase text and separate punctuation for BLEU evaluation.
    """

    text = text.lower()

    text = re.sub(
        r"([.,!?;:()])",
        r" \1 ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text.split()


# ============================================================
# Load VRSBench reference captions
# ============================================================

def load_references():

    print("=" * 60)
    print("Loading VRSBench reference captions...")
    print("=" * 60)

    with open(
        REFERENCE_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    references = {}

    for item in data:

        if not isinstance(item, dict):
            continue

        image_id = item.get("image_id")

        if image_id in IMAGE_IDS:

            ground_truth = item.get("ground_truth")

            if ground_truth:
                references[image_id] = ground_truth

    print("\nReferences found:")

    for image_id in IMAGE_IDS:

        if image_id in references:
            print("  OK:", image_id)

        else:
            print("  MISSING:", image_id)

    return references


# ============================================================
# Load generated captions
# ============================================================

def load_generated_captions():

    print("\n" + "=" * 60)
    print("Loading generated captions...")
    print("=" * 60)

    with open(
        GENERATED_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        text = f.read()

    generated = {}

    lines = text.splitlines()

    current_image = None
    current_caption = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # New image entry
        if line in IMAGE_IDS:

            # Save previous caption
            if current_image is not None:

                generated[current_image] = " ".join(
                    current_caption
                ).strip()

            current_image = line
            current_caption = []

        elif current_image is not None:

            if not line.startswith("ERROR:"):
                current_caption.append(line)

    # Save final caption
    if current_image is not None:

        generated[current_image] = " ".join(
            current_caption
        ).strip()

    print("\nGenerated captions found:")

    for image_id in IMAGE_IDS:

        if image_id in generated:
            print("  OK:", image_id)

        else:
            print("  MISSING:", image_id)

    return generated


# ============================================================
# BLEU calculation
# ============================================================

def calculate_bleu(references, generated):

    print("\n" + "=" * 60)
    print("Calculating BLEU scores...")
    print("=" * 60)

    smoothing = SmoothingFunction().method1

    results = []

    corpus_references = []
    corpus_candidates = []

    for image_id in IMAGE_IDS:

        if image_id not in references:
            print(
                f"\nSkipping {image_id}: "
                "reference missing."
            )
            continue

        if image_id not in generated:
            print(
                f"\nSkipping {image_id}: "
                "generated caption missing."
            )
            continue

        reference_tokens = tokenize(
            references[image_id]
        )

        candidate_tokens = tokenize(
            generated[image_id]
        )

        # ----------------------------------------------------
        # BLEU-1
        # ----------------------------------------------------

        bleu1 = sentence_bleu(
            [reference_tokens],
            candidate_tokens,
            weights=(1.0, 0.0, 0.0, 0.0),
            smoothing_function=smoothing,
        )

        # ----------------------------------------------------
        # BLEU-2
        # ----------------------------------------------------

        bleu2 = sentence_bleu(
            [reference_tokens],
            candidate_tokens,
            weights=(0.5, 0.5, 0.0, 0.0),
            smoothing_function=smoothing,
        )

        # ----------------------------------------------------
        # BLEU-3
        # ----------------------------------------------------

        bleu3 = sentence_bleu(
            [reference_tokens],
            candidate_tokens,
            weights=(1 / 3, 1 / 3, 1 / 3, 0.0),
            smoothing_function=smoothing,
        )

        # ----------------------------------------------------
        # BLEU-4
        # ----------------------------------------------------

        bleu4 = sentence_bleu(
            [reference_tokens],
            candidate_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing,
        )

        results.append({
            "image": image_id,
            "bleu1": bleu1,
            "bleu2": bleu2,
            "bleu3": bleu3,
            "bleu4": bleu4,
        })

        corpus_references.append(
            [reference_tokens]
        )

        corpus_candidates.append(
            candidate_tokens
        )

        print("\n" + image_id)

        print(
            f"BLEU-1: {bleu1:.4f}"
        )

        print(
            f"BLEU-2: {bleu2:.4f}"
        )

        print(
            f"BLEU-3: {bleu3:.4f}"
        )

        print(
            f"BLEU-4: {bleu4:.4f}"
        )

    # --------------------------------------------------------
    # Make sure there is something to evaluate
    # --------------------------------------------------------

    if not corpus_candidates:

        raise RuntimeError(
            "No valid image/reference pairs were found."
        )

    # --------------------------------------------------------
    # Aggregate BLEU
    # --------------------------------------------------------

    corpus_bleu1 = corpus_bleu(
        corpus_references,
        corpus_candidates,
        weights=(1.0, 0.0, 0.0, 0.0),
        smoothing_function=smoothing,
    )

    corpus_bleu2 = corpus_bleu(
        corpus_references,
        corpus_candidates,
        weights=(0.5, 0.5, 0.0, 0.0),
        smoothing_function=smoothing,
    )

    corpus_bleu3 = corpus_bleu(
        corpus_references,
        corpus_candidates,
        weights=(1 / 3, 1 / 3, 1 / 3, 0.0),
        smoothing_function=smoothing,
    )

    corpus_bleu4 = corpus_bleu(
        corpus_references,
        corpus_candidates,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing,
    )

    aggregate = (
        corpus_bleu1,
        corpus_bleu2,
        corpus_bleu3,
        corpus_bleu4,
    )

    print("\n" + "=" * 60)
    print("AGGREGATE BLEU")
    print("=" * 60)

    print(f"BLEU-1: {corpus_bleu1:.4f}")
    print(f"BLEU-2: {corpus_bleu2:.4f}")
    print(f"BLEU-3: {corpus_bleu3:.4f}")
    print(f"BLEU-4: {corpus_bleu4:.4f}")

    return results, aggregate


# ============================================================
# Save results
# ============================================================

def save_results(
    results,
    aggregate,
    references,
    generated,
):

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "VRSBench Task 2 - BLEU Evaluation\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            "Evaluation set: 6 required VRSBench images\n"
        )

        f.write(
            "Reference source: VRSBench_EVAL_Cap.json\n"
        )

        f.write(
            "Metric: BLEU-1, BLEU-2, BLEU-3, BLEU-4\n"
        )

        f.write(
            "Smoothing: NLTK method1\n\n"
        )

        # ----------------------------------------------------
        # Per-image scores
        # ----------------------------------------------------

        f.write(
            "PER-IMAGE RESULTS\n"
        )

        f.write(
            "-" * 60 + "\n\n"
        )

        for result in results:

            image_id = result["image"]

            f.write(
                f"{image_id}\n"
            )

            f.write(
                f"BLEU-1: {result['bleu1']:.4f}\n"
            )

            f.write(
                f"BLEU-2: {result['bleu2']:.4f}\n"
            )

            f.write(
                f"BLEU-3: {result['bleu3']:.4f}\n"
            )

            f.write(
                f"BLEU-4: {result['bleu4']:.4f}\n"
            )

            f.write("\n")

        # ----------------------------------------------------
        # Aggregate
        # ----------------------------------------------------

        f.write(
            "AGGREGATE RESULTS\n"
        )

        f.write(
            "-" * 60 + "\n\n"
        )

        f.write(
            f"BLEU-1: {aggregate[0]:.4f}\n"
        )

        f.write(
            f"BLEU-2: {aggregate[1]:.4f}\n"
        )

        f.write(
            f"BLEU-3: {aggregate[2]:.4f}\n"
        )

        f.write(
            f"BLEU-4: {aggregate[3]:.4f}\n"
        )

        f.write("\n")

        # ----------------------------------------------------
        # Generated captions
        # ----------------------------------------------------

        f.write(
            "GENERATED CAPTIONS\n"
        )

        f.write(
            "-" * 60 + "\n\n"
        )

        for image_id in IMAGE_IDS:

            if image_id in generated:

                f.write(
                    f"{image_id}\n"
                )

                f.write(
                    generated[image_id]
                )

                f.write("\n\n")

    print("\nResults saved to:")
    print(OUTPUT_FILE)


# ============================================================
# Main
# ============================================================

def main():

    references = load_references()

    generated = load_generated_captions()

    results, aggregate = calculate_bleu(
        references,
        generated
    )

    save_results(
        results,
        aggregate,
        references,
        generated
    )

    print("\n" + "=" * 60)
    print("Evaluation finished successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()