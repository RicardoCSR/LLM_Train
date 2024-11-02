import json
from langchain_community.llms import Ollama
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from datetime import datetime

llm = Ollama(
    model="gemma2:2b",
    temperature=0.9
)

codigo_manual = "MMOE0079"
nome_produto = "HS-30"
output_file_name = "fine_tuning_dataset_MMOE0079.json"
path = "R:\\Manuais"
text_loader_kwargs = {'autodetect_encoding': True}

print("Start Loading =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
loader = DirectoryLoader(
    path,
    loader_kwargs=text_loader_kwargs,
    use_multithreading=True,
    show_progress=True
)

documents = loader.load()
print("End Loading =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
print(f"Total of documents loaded: {len(documents)}")

print("Start Splitting-Storing-Retriever =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
text_splitter = CharacterTextSplitter(
    separator=" ",
    chunk_size=512,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True
)

text_chunks = text_splitter.split_documents(documents)
print(f"Split {len(documents)} documents into {len(text_chunks)} chunks.")

embeddings = HuggingFaceEmbeddings()
knowledge_base = FAISS.from_documents(
    text_chunks,
    embeddings
)
print("End Splitting-Storing-Retriever =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

fine_tuning_data = []
instruction_list = [
    "Resuma o conteúdo do manual a seguir, focando nas instruções de operação.",
    "Extraia as principais informações deste manual.",
    "Converta as instruções do seguinte manual para um formato mais formal."
]

for i, chunk in enumerate(text_chunks):
    input_text = chunk.page_content.strip()

    if not input_text:
        continue

    context = f"Este é um trecho do manual {codigo_manual} do produto {nome_produto}."
    formatted_input = f"{context}\n\nInput:\n{input_text}"

    for instruction in instruction_list:
        prompt = f"{instruction}\nPor favor, responda em português do Brasil.\n\n{formatted_input}"
        print(f"Processing chunk {i + 1}/{len(text_chunks)} - Instruction: {instruction}")
        try:
            output_text = llm(prompt)
        except Exception as e:
            print(f"Error processing chunk {i + 1}: {e}")
            continue

        text_concat = f"### Instruction: {instruction}\n### Input: {formatted_input}\n### Response: {output_text}"

        fine_tuning_data.append({
            "instruction": instruction,
            "input": formatted_input,
            "output": output_text,
            "text": text_concat
        })

with open(output_file_name, 'w', encoding='utf-8') as f:
    json.dump(fine_tuning_data, f, ensure_ascii=False, indent=4)

print("Dataset de fine-tuning gerado com sucesso!")
