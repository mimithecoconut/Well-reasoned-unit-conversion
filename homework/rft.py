from .base_llm import BaseLLM
from .sft import test_model, tokenize, TokenizedDataset
from .data import Dataset, benchmark
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer, TrainingArguments
import json


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm

def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    try:
      ans_float = float(answer)
      answer_str = f"<answer>{round(ans_float, 4)}</answer>"
    except Exception:
      answer_str = f"<answer>{answer}</answer>"
    
    return {"question": prompt, "answer": answer_str}

class RFTDataset:
    """Simple wrapper for RFT json entries"""

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q, gold, reasoning = self.data[idx]
        # reasoning already contains <answer>
        return tokenize(self.tokenizer, q, reasoning)


def train_model(
    output_dir: str = "homework/rft_model",
    **kwargs,
):  
    rft_path = "data/rft.json"
    base = BaseLLM()
    tokenizer = base.tokenizer

    # Load RFT data
    with open(rft_path, "r") as f:
        raw = json.load(f)

    train_ds = RFTDataset(raw, tokenizer)

    # Larger LoRA rank for reasoning tasks
    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    base.model = get_peft_model(base.model, lora_config)

    if base.device == "cuda":
        base.model.enable_input_require_grads()

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        per_device_train_batch_size=32,
        num_train_epochs=8,
        learning_rate=5e-4,
        gradient_checkpointing=True,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=base.model,
        args=training_args,
        train_dataset=train_ds,
    )

    trainer.train()
    trainer.save_model(output_dir)

    test_model(output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
