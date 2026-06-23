from langchain_core.prompts import PromptTemplate

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
