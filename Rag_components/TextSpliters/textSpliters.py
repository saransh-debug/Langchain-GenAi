from langchain_text_splitters import CharacterTextSplitter

# Character Text Splitter is a class that splits text into chunks based on character count. It is useful for processing large documents by breaking them down into smaller, manageable pieces. The CharacterTextSplitter can be configured with parameters such as chunk size and overlap to control how the text is split. 
# negetive points of this text splitter include the potential for splitting sentences or words inappropriately, which can lead to loss of context or meaning. Additionally, it may not be suitable for all types of text, especially those that require semantic understanding or context preservation.

spltr = CharacterTextSplitter(
    separator=" ",
    chunk_size = 10 , 
    chunk_overlap = 2 # chunk overlap is the number of characters that will be repeated in the next chunk to maintain context. For example, if the chunk size is 10 and the overlap is 2, the first chunk will contain characters 0-9, and the second chunk will start from character 8 (overlapping the last 2 characters of the first chunk). This helps to preserve context between chunks, especially when processing text for tasks like natural language understanding or information retrieval.
)

query = " a random text that needs to be split into smaller chunks for processing. This text is just an example to demonstrate how the CharacterTextSplitter works. It will be divided into segments based on the specified chunk size and overlap settings."

result = spltr.split_text(query)

print(result)