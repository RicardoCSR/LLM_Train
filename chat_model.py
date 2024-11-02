from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Caminho para o modelo
model_path = "R:/Udemy_Study/LLM_Chatbot/meu_modelo_treinado/model.safetensor"  # Ajuste conforme necessário

# Carregar o tokenizer
tokenizer = T5Tokenizer.from_pretrained(
    "unicamp-dl/ptt5-base-portuguese-vocab")  # Ou o tokenizer correspondente ao seu modelo

# Carregar o modelo a partir do arquivo .safetensor
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Verificar se uma GPU está disponível e mover o modelo para a GPU se possível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def gerar_resposta(input_text):
    # Formatar a entrada para dar contexto ao modelo
    formatted_input = f"Responda: {input_text}"  # Usar "Responda:" como prompt

    # Tokenizar a entrada e mover para o dispositivo correto
    input_ids = tokenizer.encode(formatted_input, return_tensors="pt").to(device)

    # Gerar a resposta com parâmetros ajustados
    output_ids = model.generate(
        input_ids,
        max_length=150,
        num_beams=5,
        temperature=0.7,  # Ajuste a temperatura para controlar a aleatoriedade
        top_k=50,
        top_p=0.95,
        do_sample=True,
        early_stopping=True
    )

    # Decodificar a resposta
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response


def main():
    print("Chatbot: Olá! Estou aqui para conversar. Digite '/sair' para encerrar.")

    while True:
        user_input = input("Você: ")

        if user_input.lower() == '/sair':
            print("Chatbot: Até logo!")
            break

        response = gerar_resposta(user_input)
        print(f"Chatbot: {response}")


if __name__ == "__main__":
    main()
