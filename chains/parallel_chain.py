from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

template1 = PromptTemplate(
    template=""" create some notes about \n {text}""" , 
    input_variables=['text']
)

template2 = PromptTemplate(
    template=""" list down questions from this {text} for a quiz.""" , 
    input_variables=['text']
)

template3 = PromptTemplate(template=""" merge the notes->{notes} and questions-> {questions} \n , in a single document.""" , 
                           input_variables=["notes" , "questions"])

model1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

model2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


parser = StrOutputParser()


parallel_chain = RunnableParallel({
    "notes":template1 | model1 | parser , 
    "questions": template2 | model2 | parser
})

sequential_chain = template3 | model2 | parser

chain = parallel_chain | sequential_chain 

text = """Vectors are fundamental mathematical objects used to represent quantities that have both magnitude and direction. Unlike scalars, which only describe magnitude (such as temperature or mass), vectors provide a richer way to model real-world phenomena like velocity, force, and displacement.

A vector is often written as an ordered pair or tuple, for example 
(
𝑥
,
𝑦
)
 in two dimensions or 
(
𝑥
,
𝑦
,
𝑧
)
 in three dimensions. Graphically, it can be represented as an arrow pointing from one point to another, where the length of the arrow corresponds to the magnitude and the orientation corresponds to the direction.

One of the most important operations with vectors is addition. When two vectors are added, their magnitudes and directions combine according to the parallelogram rule or the triangle rule. Similarly, vectors can be subtracted to find relative displacement. Another key operation is scalar multiplication, which stretches or shrinks a vector without changing its direction.

Vectors also allow us to compute quantities like the dot product and cross product. The dot product measures how much two vectors align, producing a scalar value, while the cross product (in three dimensions) produces another vector perpendicular to the original two. These operations are essential in physics, engineering, and computer graphics.

Applications of vectors are vast. In physics, they describe forces acting on objects. In computer science, vectors are used in graphics rendering, machine learning, and data representation. Navigation systems rely on vectors to calculate directions and distances. Even in everyday life, when you walk a certain distance north and then east, you are essentially performing vector addition.

In summary, vectors are powerful tools that bridge abstract mathematics with practical applications. Their ability to capture both magnitude and direction makes them indispensable in understanding and modeling the world around us."""
result  = chain.invoke({"text":text})
print(result)
chain.get_graph().print_ascii()
