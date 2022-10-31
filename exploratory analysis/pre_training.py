from transformers import Trainer, TrainingArguments
from transformers import BertTokenizer, LineByLineTextDataset, BertModel, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling

model_path = './ckp/pretraining/checkpoint-12500'

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Convert data into required format
dataset= LineByLineTextDataset(tokenizer = tokenizer, file_path = './unlabelled_articles_17K/pretraining_data.txt', block_size = 256)
print('Dataset Length:', len(dataset))

model = BertForMaskedLM.from_pretrained(model_path)
print('Number of parameters:', model.num_parameters())


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Defining training configuration\
training_args = TrainingArguments(
    output_dir='./ckp/pretraining_2/',
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_device_train_batch_size=32,
    save_steps=2_500,
    save_total_limit=5,
    gradient_accumulation_steps = 2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Perfrom pre-training and save the model
trainer.train()
trainer.save_model('./ckp/pretrained_bert-base/')