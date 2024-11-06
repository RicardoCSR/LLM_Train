# ----------------------- LIBRARY --------------------------
import os
import pandas as pd
from PyPDF2 import PdfReader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from transformers import AutoModel
import torch
from datetime import datetime
from huggingface_hub import HfApi

# ----------------------- SETTING --------------------------
os.environ["PYTORCH_TRANSFORMERS_CACHE"] = "R:/LLM/Models"  # Local de cache dos modelos hf
access_token = "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"

main_model = "google/gemma-2-9b"  # Modelo para tokenização e interpretação de dados
verification_model = "google/flan-t5-large"  # Modelo para análise de verificação
file_folder = r"D:\Manual"  #"C:\Users\Ricardo\Desktop\Manual RD"     # Endereço local da pasta com arquivos
max_length = 512    # Tamanho dos Tokenizer
test_size = 0.2     # Porcentagem de dados utilizados para Análise de Verificação

custom_column = [     # Colunas para Dataset
    "Nome do Produto",
    "Descrição",
    "Categoria",
    "Parâmetros Monitorados",
    "Características Principais",
    "Código do Produto"
]

'''
1. **Nome do Produto**: (ex: Monitor Multiparamétrico RD10)
2. **Descrição**: (ex: Monitor com tela de 10" para monitoramento de pacientes.)
3. **Categoria**: (ex: Monitores Multiparamétricos)
4. **Parâmetros Monitorados**: (ex: ECG, Pressão Arterial, Oximetria)
5. **Características Principais**: (ex: Tela colorida TFT, 72 horas de tendência, gestão inteligente de alarmes)
6. **Código do Produto**: (ex: PMRD1001STD)
'''

# ----------------------- FUNÇÃO PARA EXTRAIR TEXTO PDF --------------------------
def extract_text_from_pdf(pdf_path):
    text = []
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    except Exception as e:
        print(f"Erro ao extrair texto de {pdf_path}: {e}")
    return " ".join(text)  # Retorna o texto como uma única string

# ----------------------- FUNÇÃO PARA VERIFICAÇÃO E CORREÇÃO CRUZADA --------------------------
def verify_and_correct(text, model1_output, model2):
    # Prompt para verificação do Modelo 2 em caso de divergência
    verification_prompt = (
        f"O Modelo 1 identificou '{model1_output}' no texto: '{text}'. "
        "Verifique se este é o resultado correto. Responda com Y (sim) ou N (não) e sugira a correção, se necessário."
    )
    model2_response = model2(verification_prompt, max_length=50)[0]["generated_text"]

    if "Y" in model2_response:
        return model1_output
    else:
        correction_suggestion = model2_response.split(":")[-1].strip()
        return correction_suggestion

# ----------------------- FUNÇÃO PARA PROCESSAR TEXTO COM LLM --------------------------
def process_text_with_llm(text, columns=None):
    # Inicializar pipelines dos modelos
    llm_pipeline_1 = pipeline("text-generation", model=main_model, tokenizer=main_model)
    llm_pipeline_2 = pipeline("text-generation", model=verification_model, tokenizer=verification_model)

    if columns is None:
        # Solicitar ao LLM a estrutura ideal de colunas
        columns_prompt = "Identifique as colunas adequadas para este dataset de produtos a partir do texto."
        suggested_columns = llm_pipeline_1(columns_prompt, max_length=100)[0]["generated_text"]
        columns = [col.strip() for col in suggested_columns.split(",")]
        print("Colunas sugeridas:", columns)

    # Estrutura de prompt para preencher as colunas com dados do produto
    extraction_prompt = f"""
    A partir do texto a seguir, extraia as informações relevantes para as colunas {columns}:
    Texto: {text}
    """
    # Modelo 1 faz a extração inicial
    extracted_info = llm_pipeline_1(extraction_prompt, max_length=300)[0]["generated_text"]

    # Converte o texto em um dicionário usando as colunas fornecidas ou sugeridas
    product_data = {}
    for col in columns:
        # Localiza os dados de cada coluna
        start = extracted_info.find(col) + len(col) + 1
        end = extracted_info.find("\n", start)
        extracted_value = extracted_info[start:end].strip()

        # Verifica e corrige o valor extraído
        corrected_value = verify_and_correct(text, extracted_value, llm_pipeline_2)
        product_data[col] = corrected_value

    return product_data

# ----------------------- MAIN --------------------------
api = HfApi(token=access_token)

print("Start Loading Files =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

all_products = []

for filename in os.listdir(file_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(file_folder, filename)
        print(f"Extraindo texto de: {pdf_path}")

        extracted_text = extract_text_from_pdf(pdf_path)
        if extracted_text:
            # Processa o texto extraído com o modelo de linguagem para obter os dados do produto
            product_info = process_text_with_llm(extracted_text, columns=custom_column)
            all_products.append(product_info)
        else:
            print(f"Arquivo {filename} está vazio ou não contém texto.")

# Criar um DataFrame a partir da lista de produtos coletados
df = pd.DataFrame(all_products)
print("End Loading Files =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

# Verificar as primeiras linhas do DataFrame
print("Primeiras linhas do DataFrame:")
print(df.head())

# Converter o DataFrame em um Dataset Hugging Face
dataset = Dataset.from_pandas(df)

# Inicializar o tokenizer e o modelo
print("Starting Tokenize Files =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
tokenizer = AutoTokenizer.from_pretrained(main_model)
model = AutoModelForCausalLM.from_pretrained(verification_model)

# Definir a função de pré-processamento
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples['texto'],
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    model_inputs['labels'] = model_inputs['input_ids']
    return model_inputs

# Aplicar a função ao dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Dividir o dataset em conjuntos de treinamento e validação
train_test_split = tokenized_dataset.train_test_split(test_size=test_size)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Salvar os datasets
train_dataset.save_to_disk("train_dataset")
eval_dataset.save_to_disk("eval_dataset")
print("End Tokenize Files =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

# Exemplo dos datasets processados
print("Exemplo do conjunto de treinamento tokenizado:")
print(train_dataset[0])

print("Exemplo do conjunto de validação tokenizado:")
print(eval_dataset[0])




'''
DATASET MODELO

Nome do Produto                         - Monitor Multiparamêtrico RD10
Descrição                               - Monitor com tela de "10" para monitoramento de pacientes
Categoria                               - Monitores Multiparamêtrico
Parâmetros Monitorados                  - ECG, Pressão Arterial, Oximetria, Temperatura, Capnografia
Características Principais              - Tela colorida TFT, 72 horas de tendência, gestão inteligente de alarmes, bateria recarregável.
Código do Produto                       - PMRD1001STD

----------------------------------------------------------------------------------

| **Nome do Produto** | **Descrição** | **Categoria** | **Parâmetros Monitorados**  | **Características Principais** | **Código do Produto** |
|---------------------|---------------|---------------|-----------------------------|--------------------------------|-----------------------|
|                     |               |               |                             |                                |                       |
|                     |               |               |                             |                                |                       |
|                     |               |               |                             |                                |                       |

'''