# ----------------------- IMPORTS --------------------------
import os
import pandas as pd
from PyPDF2 import PdfReader
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm

# ----------------------- CONFIGURAÇÕES --------------------------
file_folder = r"D:\Manual"  # Altere para o diretório dos PDFs
output_csv = "dataset_produtos.csv"
custom_columns = ["Nome do Produto", "Descrição", "Categoria",
                  "Parâmetros Monitorados", "Características Principais", "Código do Produto"]

# Carregar o pipeline de NER e tokenizador
model_name = "pablocosta/ner-bert-base-portuguese-cased"
ner_pipeline = pipeline("ner", model=model_name, tokenizer=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
max_token_length = 512


# ----------------------- FUNÇÃO PARA EXTRAIR TEXTO DE PDF --------------------------
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


# ----------------------- FUNÇÃO PARA DIVIDIR TEXTO EM PARTES MENORES --------------------------
def split_text_into_chunks(text, max_length=512):
    tokens = tokenizer(text, truncation=False, return_tensors="pt")["input_ids"][0]
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]


# ----------------------- FUNÇÃO PARA EXTRAIR INFORMAÇÕES COM NER --------------------------
def extract_info_with_ner(text):
    # Divide o texto em chunks se exceder o tamanho máximo permitido
    text_chunks = split_text_into_chunks(text, max_length=max_token_length)
    extracted_data = {col: "" for col in custom_columns}

    # Processa cada chunk separadamente
    for chunk in text_chunks:
        ner_results = ner_pipeline(chunk)
        entity_mapping = {
            "PRODUCT_NAME": "Nome do Produto",
            "DESCRIPTION": "Descrição",
            "CATEGORY": "Categoria",
            "PARAMETERS": "Parâmetros Monitorados",
            "FEATURES": "Características Principais",
            "PRODUCT_CODE": "Código do Produto"
        }

        # Preenche o dicionário com os resultados do NER
        for entity in ner_results:
            entity_type = entity['entity']  # Tipo de entidade
            entity_text = entity['word']  # Texto da entidade
            if entity_type in entity_mapping:
                column = entity_mapping[entity_type]
                extracted_data[column] += entity_text + " "

    # Limpa os valores de cada campo
    return {key: value.strip() for key, value in extracted_data.items()}


# ----------------------- FUNÇÃO PRINCIPAL PARA PROCESSAR ARQUIVOS E GERAR DATASET --------------------------
def process_pdfs_and_generate_dataset():
    all_products = []

    for filename in tqdm(os.listdir(file_folder), desc="Processando PDFs"):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(file_folder, filename)
            print(f"Extraindo texto de: {pdf_path}")
            extracted_text = extract_text_from_pdf(pdf_path)
            if extracted_text:
                product_info = extract_info_with_ner(extracted_text)
                all_products.append(product_info)
            else:
                print(f"Arquivo {filename} está vazio ou não contém texto.")

    df = pd.DataFrame(all_products, columns=custom_columns)
    df.to_csv(output_csv, index=False)
    print(f"Dataset salvo como {output_csv}")

    print("Primeiras linhas do DataFrame:")
    print(df.head())


# ----------------------- EXECUÇÃO --------------------------
if __name__ == "__main__":
    process_pdfs_and_generate_dataset()
