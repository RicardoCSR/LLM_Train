import os
import pandas as pd
from PyPDF2 import PdfReader
from datasets import Dataset
from transformers import AutoTokenizer

# Função para extrair texto de um PDF
def extract_text_from_pdf(pdf_path):
    text = []
    reader = PdfReader(pdf_path)

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:  # Verifica se a página contém texto
            text.append(page_text)

    return text


# Caminho para a pasta que contém os PDFs
caminho_pasta = r"C:\Users\Ricardo\Desktop\Manual RD"  # Altere para o caminho da sua pasta

# Lista para armazenar textos extraídos
all_texts = []

# Iterar sobre todos os arquivos na pasta
for filename in os.listdir(caminho_pasta):
    if filename.endswith(".pdf"):  # Verifica se o arquivo é um PDF
        pdf_path = os.path.join(caminho_pasta, filename)
        print(f"Extraindo texto de: {pdf_path}")
        extracted_text = extract_text_from_pdf(pdf_path)
        all_texts.extend(extracted_text)  # Adiciona o texto extraído à lista

# Criar um DataFrame a partir do texto extraído
df = pd.DataFrame({'texto': all_texts})

# Verificar as primeiras linhas do DataFrame
print("Primeiras linhas do DataFrame:")
print(df.head())

# Converter o DataFrame em um Dataset Hugging Face
dataset = Dataset.from_pandas(df)

# Inicializar o tokenizer
tokenizer = AutoTokenizer.from_pretrained("unicamp-dl/ptt5-base-portuguese-vocab")


# Definir a função de pré-processamento
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples['texto'],  # Nome da coluna que contém os textos
        padding="max_length",
        truncation=True,
        max_length=512  # Defina conforme necessário
    )

    # Para T5, precisamos também dos rótulos (decoder_input_ids)
    labels = tokenizer(
        examples['texto'],  # Aqui você deve usar o texto esperado (rótulo)
        padding="max_length",
        truncation=True,
        max_length=512  # Defina conforme necessário
    )

    model_inputs['labels'] = labels['input_ids']  # Adiciona as labels ao dicionário

    return model_inputs


# Aplicar a função ao dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Dividir o dataset em conjuntos de treinamento e validação (80% treino, 20% validação)
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Salvar os datasets em formato Parquet (opcional)
train_dataset.save_to_disk("train_dataset")
eval_dataset.save_to_disk("eval_dataset")

# Verificar os resultados tokenizados
print("Exemplo do conjunto de treinamento tokenizado:")
print(train_dataset[0])  # Exibir o primeiro exemplo do conjunto de treinamento tokenizado

print("Exemplo do conjunto de validação tokenizado:")
print(eval_dataset[0])  # Exibir o primeiro exemplo do conjunto de validação tokenizado
