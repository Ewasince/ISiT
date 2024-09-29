# K-neighbours.


## Preview

[> Jupiter file with list code](recsys.ipynb)

У нас есть модель ввода:
```python
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
```


Образование вектора пользовательских данных

```python
class RecSys:
    @staticmethod
    def to_df(user_model: UserInput):
        return pandas.DataFrame(
            dict(
                gender=[user_model.gender],
                age=[user_model.age],
                heatlfy=[user_model.heatlfy],
                smoke=[user_model.smoke],
                stress=[user_model.stress],
                sleep_well=[user_model.sleep_well],
                chronus=[user_model.chronus],
                wake_up=[user_model.wake_up],
                sleep_time=[user_model.sleep_time],
                coffe_near=[user_model.coffe_near],
                gourmet=[user_model.gourmet],
                office=[user_model.office],
                home_seater=[user_model.home_seater],
                ill=[user_model.ill],
            )
        )
```

выбрано радиально-базисное расстояние, потому что RBF хорошо работает для сложных нелинейных зависимостей, когда 
бинарные и количественные признаки могут взаимодействовать
```python
diffs = rbf_kernel(target_vectors, user_vector)
result_df = pandas.DataFrame(diffs, columns=['similarity_rate'])

```