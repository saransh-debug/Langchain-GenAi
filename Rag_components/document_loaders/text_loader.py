from langchain_community.document_loaders import TextLoader


loader = TextLoader(r"C:\Users\HP\OneDrive\Desktop\langchain\Rag_components\document_loaders\random.txt") 
result = loader.load()
print(result)                                                         