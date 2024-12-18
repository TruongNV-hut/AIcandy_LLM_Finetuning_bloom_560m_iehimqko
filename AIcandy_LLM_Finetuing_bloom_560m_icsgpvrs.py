"""

@author:  AIcandy 
@website: aicandy.vn

"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
import transformers


computation_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Model base path
original_model_path = "models/bloom-560m"

# Create tokenizer
model_tokenizer = AutoTokenizer.from_pretrained(original_model_path)

# Load model
language_model = AutoModelForCausalLM.from_pretrained(original_model_path, trust_remote_code=True)
language_model = language_model.to(computation_device)

# Gradient checkpointing (only use if RAM reduction is needed)
language_model.gradient_checkpointing_enable()

# Prepare model for k-bit training
language_model = prepare_model_for_kbit_training(language_model)

# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
language_model = get_peft_model(language_model, lora_config)

# Load data
training_dataset = load_dataset("json", data_files="datasets/*.json", split='train')
training_dataset = training_dataset.map(
    lambda samples: {
        "input_text": "Human: " + samples["question"] + " Assistant: " + samples["answer"]
    },
    remove_columns=['question', 'answer']
)
training_dataset = training_dataset.map(lambda examples: model_tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=512), batched=True)

# Ensure padding token
model_tokenizer.pad_token = model_tokenizer.eos_token

# Trainer with detailed logging
model_trainer = transformers.Trainer(
    model=language_model,
    train_dataset=training_dataset,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        logging_steps=1,  # Log after each step
        output_dir="outputs",
        optim="adamw_hf",
        
        # Add detailed logging parameters
        logging_dir="./logs",  # Log directory
        logging_strategy="steps",
        logging_first_step=True,  # Log from the first step
        report_to="all",  # Report to all tracking systems

        # Add more detailed parameters
        disable_tqdm=False,  # Show progress bar
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(model_tokenizer, mlm=False),
)

language_model.config.use_cache = False
# Perform training with detailed output
try:
    training_result = model_trainer.train()
    print("Training completed successfully!")
    print(f"Final training loss: {training_result.training_loss}")
except Exception as e:
    print(f"An error occurred during training: {e}")

# Adapter model save path
adapter_model_path = "models/adapter-model"

# Save model after fine-tuning
model_trainer.save_model(adapter_model_path)

# Merge adapter with original model
base_language_model = AutoModelForCausalLM.from_pretrained(
    original_model_path,
    return_dict=True,
    trust_remote_code=True
)
merged_model = PeftModel.from_pretrained(base_language_model, adapter_model_path).to(computation_device)
merged_model = merged_model.merge_and_unload()

# Save merged model
merged_model_path = "models/merged-model"
merged_model.save_pretrained(merged_model_path)
model_tokenizer.save_pretrained(merged_model_path)



# ------------------------- for test -------------
# Generate text
merged_model_path = "models/merged-model"
test_model = AutoModelForCausalLM.from_pretrained(merged_model_path).to(computation_device)
test_tokenizer = AutoTokenizer.from_pretrained(merged_model_path)

test_prompt = "Human: Trí tuệ nhân tạo là gì? Assistant: "
input_sequence = test_tokenizer.encode(test_prompt, return_tensors='pt').to(computation_device)

generated_outputs = test_model.generate(
    input_sequence,
    pad_token_id=50256,
    do_sample=True,
    max_length=100,
    temperature=0.8,
    early_stopping=False,
    no_repeat_ngram_size=2,
    num_return_sequences=1
)

for i, generated_output in enumerate(generated_outputs):
    print(f">> Generated text {i + 1}\n\n{test_tokenizer.decode(generated_output.tolist())}")
