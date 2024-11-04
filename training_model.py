from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datetime import datetime

# Carregar os datasets salvos
print("Starting Dataset =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
train_dataset = load_from_disk("train_dataset")
eval_dataset = load_from_disk("eval_dataset")

print("End Dataset =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
print("Starting Tokenizer =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
data = "MonitorRD12-15"

# Nome do modelo treinado
training_model_name = f"{model_name}_treinado_{data}"
output_directory = './results'  # Corrigido para ser um caminho relativo

# Inicializar o tokenizer e o modelo
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)  # Usar modelo adequado

# Configura os argumentos de treinamento
'''
    2 to 8GB VRAM, 
    4 to 16GB VRAM, 
    8 to 24GB VRAM, 
    16to 32GB VRAM
'''
training_args = TrainingArguments(
    output_dir = output_directory,
    evaluation_strategy = "epoch",  # Avalia no final de cada época
    learning_rate = 3e-5,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    num_train_epochs = 5,
    weight_decay = 0.01,
    logging_dir='./logs',  # Diretório para logs
    logging_steps = 10,
    save_total_limit = 2,
)

# Cria um Trainer para gerenciar o treinamento do modelo
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
)

# Iniciar o treinamento do modelo
trainer.train()

# Salvar o modelo treinado e tokenizer para uso futuro
print("End Tokenizer =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
model.save_pretrained(training_model_name)
tokenizer.save_pretrained(training_model_name)

print(f"Modelo salvo como: {training_model_name}, no endereço: {output_directory}")
print("Modelo treinado e salvo com sucesso!")
