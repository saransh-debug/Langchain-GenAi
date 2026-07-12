# from langchain_community.document_loaders import PyPDFLoader

# loader = PyPDFLoader("Rag_components/document_loaders/saransh_main.pdf")
# docs = loader.load()
# print(docs[1].metadata)
# print(docs[1].page_content)
from langchain_community.document_loaders import UnstructuredPDFLoader ,PyPDFLoader

loader = PyPDFLoader(r"Rag_components\document_loaders\ForwardInvoice_ORD54043623914.pdf")
docs = loader.load()
print(docs)