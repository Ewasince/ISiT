import pandas

from models import UserInput
from utils import RecSys

if __name__ == '__main__':
    file = pandas.read_csv("survey.csv", sep=";")
    r = RecSys(df=file, n_neighbour=1)
    r.fit()
    input_data_1 = UserInput(
        gender="Мужчина",
        age=21,
        heatlfy=85,
        smoke="Нет",
        stress=70,
        sleep_well="Да",
        chronus="Жаворонок",
        wake_up="6:00",
        sleep_time=8,
        coffe_near="Да",
        gourmet="Нет",
        office="Нет",
        home_seater="Нет",
        ill="Нет",
    )
    result = r.predict(input_data_1)
    print(result)  # Чай

    input_data_2 = UserInput(
        gender="Женщина",
        age=22,
        heatlfy=65,
        smoke="Да",
        stress=55,
        sleep_well="Да",
        chronus="Жаворонок",
        wake_up="8:00",
        sleep_time=8,
        coffe_near="Да",
        gourmet="Да",
        office="Да",
        home_seater="Нет",
        ill="Нет",
    )
    result = r.predict(input_data_2)
    print(result)  # Кофе
