import os
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

IMAGE_DIR = "data/Images_val"

IMAGE_IDS = [
    "P0003_0002.png",
    "P0019_0002.png",
    "P0060_0004.png",
    "P0110_0017.png",
    "P0146_0005.png",
    "P0168_0009.png",
]

OUTPUT_FILE = "outputs/generated_captions.txt"


# ============================================================
# Captioning prompt
# ============================================================

PROMPT = """
Describe the satellite image using only clearly visible information.

Focus on:
- the main scene or land-use type
- important visible objects and structures
- accurate counts of objects when clearly countable
- important spatial relationships and positions

Prioritize concrete visual evidence. Do not speculate about unclear
objects or their purpose. Do not add information that is not visually
supported.

Do not mention image resolution, image source, Google Earth, or
uncertain possibilities.

Write one concise paragraph of about 50–80 words.
Return only the caption.
"""


# ============================================================
# 4-bit quantization configuration
# ============================================================

print("=" * 60)
print("Loading Qwen2.5-VL-7B-Instruct in 4-bit mode...")
print("=" * 60)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# Load model
# ============================================================

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    min_pixels=256 * 28 * 28,
    max_pixels=768 * 28 * 28,
)

print("Model loaded successfully.")
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

    print(
        "GPU memory allocated:",
        round(torch.cuda.memory_allocated() / 1024**3, 2),
        "GB"
    )


# ============================================================
# Generate caption
# ============================================================

def generate_caption(image_path):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": PROMPT,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move input tensors to the model's execution device
    inputs = inputs.to(model.device)

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
        )

    # Remove prompt tokens from generated sequence
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids
        in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0].strip()


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs("outputs", exist_ok=True)

    results = []

    for image_id in IMAGE_IDS:

        image_path = os.path.join(
            IMAGE_DIR,
            image_id
        )

        print("\n" + "=" * 60)
        print("IMAGE:", image_id)
        print("=" * 60)

        if not os.path.exists(image_path):
            print("ERROR: Image not found:")
            print(image_path)
            continue

        try:

            caption = generate_caption(image_path)

            print("\nGENERATED CAPTION:")
            print(caption)

            results.append(
                f"{image_id}\n{caption}\n"
            )

        except Exception as e:

            print("\nERROR:", e)

            results.append(
                f"{image_id}\nERROR: {e}\n"
            )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("\n".join(results))

    print("\n" + "=" * 60)
    print("Finished.")
    print("Results saved to:", OUTPUT_FILE)
    print("=" * 60)


if __name__ == "__main__":
    main()