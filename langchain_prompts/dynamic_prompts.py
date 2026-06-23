from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st 
from langchain_core.prompts import PromptTemplate
load_dotenv()

st.header("Research tool :") # gives the header of the page 

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash") # to select the model 

topic = st.selectbox( # dropdown box in the streamlit 
    "Select the topic",
    [
        "Artificial Intelligence",
        "Machine Learning",
        "Python Programming",
        "Data Science",
        "Web Development",
        "Cyber Security",
        "Cloud Computing",
        "Blockchain",
        "Internet of Things",
        "Generative AI"
    ]
)

style = st.selectbox(
    "Select the style",
    [
        "Beginner Friendly",
        "Professional",
        "Technical",
        "Creative",
        "Humorous",
        "Storytelling",
        "Academic",
        "Conversational"
    ]
)

length = st.selectbox(
    "Select the length",
    [
        "50 words",
        "100 words",
        "200 words",
        "300 words",
        "500 words"
    ]
)

template = PromptTemplate( # a class in langchain that is used to give dynamic prompts 
    template="""
    Write an essay on the topic {topic} , 
    and make sure its writing style should be {style} of length {length}.
    in case no information is available , return the output as 'No information available about this topic .'
    """ , 
    input_variables = ["topic", # to tell the variables name to the template 
"style",
"length",
    ] , 
    validate_template=True # validates the input variables that are required and how many of them we had recieved .
)

# prompt = template.invoke({ # to send the variables to the static template 
#     "topic":f"{topic}",
# "style":f"{style}",
# "length":f"{length}"
# })




if st.button("summarize"):
    chain = template | model # chaining saves two invokes call and only a single invoke does the job . 
    result = chain.invoke({
        "topic":f"{topic}",
        "style":f"{style}",
        "length":f"{length}"
    })
    st.write(result.content)