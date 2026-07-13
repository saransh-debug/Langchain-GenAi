from langchain_text_splitters import RecursiveCharacterTextSplitter

# Recursive Character Text Splitter is a class that splits text into chunks based on a list of characters. It is useful for processing large documents by breaking them down into smaller, manageable pieces. The RecursiveCharacterTextSplitter can be configured with parameters such as chunk size and overlap to control how the text is split.
# in recursive text splitter the splitting is done recursively, meaning that if a chunk is too large, it will be further split into smaller chunks until the desired chunk size (for eg 50) is achieved. This allows for more flexible and efficient text processing, especially when dealing with complex or lengthy documents.

spltr = RecursiveCharacterTextSplitter(
    chunk_size = 50 , 
    chunk_overlap = 15 # chunk overlap is the number of characters that will be repeated in the next chunk to maintain context. For example, if the chunk size is 10 and the overlap is 2, the first chunk will contain characters 0-9, and the second chunk will start from character 8 (overlapping the last 2 characters of the first chunk). This helps to preserve context between chunks, especially when processing text for tasks like natural language understanding or information retrieval.
)

query = " a random text that needs to be split into smaller chunks for processing. This text is just an example to demonstrate how the RecursiveCharacterTextSplitter works. It will be divided into segments based on the specified chunk size and overlap settings."

result = spltr.split_text(query)

print(result)