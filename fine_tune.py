from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

config = GPT2Config.from_pretrained("gpt2", output_hidden_states=False)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", config=config)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.save_pretrained("your_output_directory")

model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="latin_tokenizer.json",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

training_args = TrainingArguments(
    output_dir="your_output_directory",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    logging_steps=1000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()
trainer.save_model("your_output_directory")
