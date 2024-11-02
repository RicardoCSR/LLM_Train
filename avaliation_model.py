from transformers import T5Tokenizer, T5ForConditionalGeneration

# Carregar seu modelo treinado
tokenizer = T5Tokenizer.from_pretrained("meu_modelo_treinado")
model = T5ForConditionalGeneration.from_pretrained("meu_modelo_treinado")

def gerar_resposta(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_ids = model.generate(input_ids)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# Teste com uma entrada
print(gerar_resposta("Qual é a capital da França?"))
