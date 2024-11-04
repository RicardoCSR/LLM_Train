import torch
from transformers import AutoTokenizer, pipeline

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Carrega o tokenizer e o pipeline
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.float16, device_map="auto")

#Prompt
messages = [
    {"role": "system", "content": "Você é uma IA chamada de Alê, muito amigável e prestativo, respondendo a todas as perguntas sempre com certeza, jamais passando qualquer informação incorreta."},
    {"role": "user", "content": "Qual a capital do Brasil?"}
]

formatted_prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

outputs = pipe(formatted_prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

print(outputs[0]["generated_text"])
