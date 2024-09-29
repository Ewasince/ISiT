from pydantic import BaseModel


class UserInput(BaseModel):
    gender: str
    age: float
    heatlfy: float
    smoke: str
    stress: float
    sleep_well: str
    chronus: str
    wake_up: str
    sleep_time: float
    coffe_near: str
    gourmet: str
    office: str
    home_seater: str
    ill: str
