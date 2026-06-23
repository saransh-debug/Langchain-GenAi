from pydantic import BaseModel , EmailStr  , Field
from typing import Optional


class Student(BaseModel):
    name : str
    lastName : str = "bhardwaj" # default value 
    age : Optional[str] = None
    gmail:EmailStr
    cgpa : float = Field(gt=0 , lt=10)



new_st = {"name":"saran" ,"gmail": "abc@gmail.com" , "cgpa":4.5}
st = Student(**new_st)
print(st)