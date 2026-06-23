from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict , Annotated , Optional
from dotenv import load_dotenv
from pydantic import BaseModel , Field
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

Review= """'ve been using this phone for about three months, and my experience has been surprisingly mixed despite its impressive specifications. The device features a flagship-grade processor, a high-refresh-rate AMOLED display, and a versatile camera system, all of which perform exceptionally well under ideal conditions. Everyday tasks such as multitasking, gaming, and media consumption feel fluid and responsive, with virtually no noticeable lag.

However, the software optimization leaves room for improvement. While the user interface is feature-rich, occasional frame drops and inconsistent app behavior can disrupt the otherwise premium experience. The battery life is generally reliable, lasting a full day with moderate to heavy usage, but standby drain appears higher than expected for a device in this price segment.

The camera system deserves particular attention. Daylight photos exhibit excellent dynamic range, sharpness, and color accuracy. Low-light performance is respectable, though image processing can sometimes be overly aggressive, resulting in unnatural textures and oversaturated colors. Video stabilization performs admirably, especially when recording in 4K.

Build quality is another highlight, with a sturdy frame and premium materials that enhance durability. Nevertheless, the device tends to heat up during extended gaming sessions and prolonged camera use.

Overall, this smartphone offers a compelling blend of performance, design, and functionality, but minor software inconsistencies prevent it from being a truly flawless flagship experience.

"""

#Schema  

class review(TypedDict):
    review : str = Field(description="give a brief summary of the review ")
    sentiment : list[str] = Field(description="give the emotions which u can find out from the review")
    keypoints : list[str]=Field(description="provide the key points present in the review")
    pros : Optional[list[str]] = Field(default=None , description="the pros present in the review")
    cons : Optional[list[str]] = Field(default=None , description="the cons present in the review")

structured_model = model.with_structured_output(review) # gives an internal prompt which orders the models to give the structured output and summary of the prompt 
result = structured_model.invoke(Review)
print(result)

