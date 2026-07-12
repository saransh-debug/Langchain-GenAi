from langchain_community.document_loaders import DirectoryLoader ,PyPDFLoader

loader = DirectoryLoader(
    path=r"C:\Users\HP\OneDrive\Desktop\langchain\Rag_components\document_loaders", 
    glob="*.pdf" , 
    loader_cls=PyPDFLoader
)

docs = loader.load()
print(docs[1].metadata)
print(docs[1].page_content)