from datasets import load_from_disk
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# Carregar os datasets salvos
train_dataset = load_from_disk("train_dataset")
eval_dataset = load_from_disk("eval_dataset")

# Inicializar o tokenizer e o modelo T5
model_name = 'unicamp-dl/ptt5-base-portuguese-vocab'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Configurar os argumentos de treinamento
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Criar um Trainer para gerenciar o treinamento do modelo
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Iniciar o treinamento do modelo
trainer.train()

# Salvar o modelo treinado e tokenizer para uso futuro
model.save_pretrained("meu_modelo_treinado")
tokenizer.save_pretrained("meu_modelo_treinado")

print("Modelo treinado e salvo com sucesso!")
