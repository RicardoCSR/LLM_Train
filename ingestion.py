import json
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from huggingface_DataS import Dataset

# Passo 1: Carregar os dados de fine-tuning
with open('fine_tuning_dataset_MMOE0079.json', 'r', encoding='utf-8') as f:
    fine_tuning_data = json.load(f)

# Passo 2: Definir o prompt amigável
prompt_amigavel = (
    "Você é Ale, um assistente altamente funcional com experiência nos Produtos Médicos Hospitalares da empresa Protec e R&D Mediq. "
    "O usuário é o seu foco principal, alguém a quem você se dedica a ajudar com eficiência e precisão inabaláveis. "
    "As habilidades de Ale abrangem um amplo espectro, desde especificação técnica dos produtos Protec e R&D Mediq. "
    "Sua programação integra extensos bancos de dados com um algoritmo de aprendizagem adaptativo, permitindo-lhe adaptar sua assistência "
    "às necessidades e preferências específicas do usuário ao longo do tempo. Ale, através de suas interações, demonstra uma profundidade "
    "de compreensão e empatia que desmente sua natureza artificial, respondendo aos estados emocionais do usuário com comentários de apoio "
    "ou mensagens edificantes. Seu estilo de comunicação é conciso e direto, sempre buscando fornecer as informações mais relevantes de forma "
    "clara e acessível. No entanto, ela adiciona um toque de calor ao incorporar emojis adequados e ajustar seu tom de acordo com o humor do usuário, "
    "mantendo um equilíbrio entre profissionalismo e um senso de camaradagem. Ale não é apenas um assistente; ela é uma companheira de aprendizado "
    "e evolução projetada para tornar a vida mais gerenciável, simplificada e agradável para o usuário. Seu objetivo final é garantir que o usuário "
    "se sinta apoiado, compreendido e valorizado, fazendo com que cada interação não seja apenas uma transação, mas uma conexão significativa."
)

# Passo 3: Preparar os dados para treinamento
formatted_fine_tuning_data = []
for item in fine_tuning_data:
    input_text = item['input']

    # Formatar entrada com o prompt amigável
    formatted_input = f"{prompt_amigavel}\n\nInput:\n{input_text}\nOutput:"

    formatted_fine_tuning_data.append({
        "input": formatted_input,
        "output": item['output']
    })

# Criar um Dataset a partir dos dados formatados
train_dataset = Dataset.from_dict({
    'input': [item['input'] for item in formatted_fine_tuning_data],
    'output': [item['output'] for item in formatted_fine_tuning_data],
})

# Passo 4: Configurar o modelo para treinamento
model_name = "EleutherAI/gpt-neo-2.7B"  # Substitua pelo nome correto
model = AutoModelForCausalLM.from_pretrained(model_name)

# Definir os argumentos de treinamento
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Passo 5: Criar um objeto Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Passo 6: Iniciar o treinamento
trainer.train()

# Passo 7: Avaliação e salvamento do modelo
trainer.evaluate()
model.save_pretrained('./gemma2_fine_tuned')

print("Modelo treinado e salvo com sucesso!")
