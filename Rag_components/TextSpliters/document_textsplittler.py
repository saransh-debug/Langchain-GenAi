from langchain_text_splitters import RecursiveCharacterTextSplitter , Language

# Document text splitter is a class that splits text into chunks based on a list of characters. It is useful for processing large documents by breaking them down into smaller, manageable pieces. The DocumentTextSplitter can be configured with parameters such as chunk size and overlap to control how the text is split.
# it is used to split the code based on the document structure, such as paragraphs, sections, or headings. This allows for more organized and meaningful text processing, especially when dealing with structured documents like articles, reports, or books.
# as code is not a normal text , it splits texts based on keywords, indentation, or other code-specific patterns. This is particularly useful for processing source code files, where maintaining the logical structure of the code is important for tasks like code analysis, refactoring, or documentation generation.
query = """
import math

class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

print(factorial(5))
print(is_prime(17))
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=100,
    chunk_overlap=20
)

result = splitter.split_text(query)
for i , chunk in enumerate(result ):
    print(f"------{i}th chunk :{chunk}")
