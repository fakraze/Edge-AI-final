# /train_qlora.py

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import torch

model_name = "meta-llama/Llama-3.2-3B-Instruct"
output_dir = "qlora-wikitext2"

# Load dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Tokenize dataset
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    # return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=192)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Load base model and prepare for QLoRA
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16
)
model = prepare_model_for_kbit_training(model)

# Apply QLoRA config
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=25,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_dir=f"{output_dir}/logs",
    save_total_limit=2,
    save_steps=500,
    # logging_steps=50,
    logging_steps=200,  # [MODIFIED] Less frequent logging
    bf16=False,
    fp16=True,  # for T4
    report_to="none",
    remove_unused_columns=False,
    # evaluation_strategy="no",
    save_strategy="epoch",
    # gradient_checkpointing=True  # [MODIFIED] Enable memory optimization
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train
trainer.train()

# Save final QLoRA model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

