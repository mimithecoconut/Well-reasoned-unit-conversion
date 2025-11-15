import json
from pathlib import Path
from tqdm import tqdm

from .cot import CoTModel
from .data import Dataset
import re


def extract_answer(text: str):
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>")[1].split("</answer>")[0].strip()
    return None


def generate_dataset(output_json: str, oversample: int = 15, temperature: float = 0.7):
    model = CoTModel()
    base_data = Dataset("train")

    results = []

    for question, gold in tqdm(base_data, desc="Generating RFT Dataset"):
        # Format the CoT chat-style prompt
        prompt = model.format_prompt(question)

        # Generate diverse reasoning chains
        outputs = model.batched_generate(
            [prompt],
            num_return_sequences=int(oversample),
            temperature=float(temperature)
        )

        chosen = None

        # Pick the first correct rollout
        for reasoning in outputs:
            pred = extract_answer(reasoning)
      
            if pred is None:
                continue

            try:
                if abs(float(pred) - float(gold)) < 1e-6:
                    chosen = reasoning
                    break
            except:
                continue

        if chosen:
            results.append([question, gold, chosen])

    # Save dataset
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} RFT examples → {output_json}")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
