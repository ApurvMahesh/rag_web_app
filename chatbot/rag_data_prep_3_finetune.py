from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Load fine-tuning data
dataset = load_dataset("json", data_files="fine_tuning_data.jsonl", split="train")

# Tokenize data
def preprocess(example):
    input_text = f"Query: {example['query']} Context: {example['context']}"
    labels = example['response']
    tokenized_input = tokenizer(input_text, max_length=512, truncation=True)
    tokenized_labels = tokenizer(labels, max_length=512, truncation=True)
    return {"input_ids": tokenized_input['input_ids'], "attention_mask": tokenized_input['attention_mask'], "labels": tokenized_labels['input_ids']}

dataset = dataset.map(preprocess, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./rag_model_finetuned",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=200,
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

# Save fine-tuned model
model.save_pretrained("./rag_model_finetuned")
tokenizer.save_pretrained("./rag_model_finetuned")
