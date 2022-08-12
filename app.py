from datetime import datetime
import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

class request_body(BaseModel): #ตั้งชื่อตัวแปร Input
    height: float
    weight: float
    age: float
    sleep: float
    somtum : int

model = pickle.load(open("svc.pickle", 'rb')) #โหลดโมเดลที่เรา Train ใน Colab
scaler = pickle.load(open("scaler.pickle", 'rb')) #โหลด Scaler 

def predict_gender(question): # Pridiction Function ที่เราสร้างจาก Colab
    data_scaled = scaler.transform([question])
    result = model.predict(data_scaled)
    if result==0:
        gender = 'ชาย'
    else:
        gender = 'หญิง'
    return f"คุณคือเพศ{gender}"

@app.get('/') # หน้าแรกของ API
def index(): 
    return {'message': 'This social time for prediction service'}
 
@app.post("/prediction") #ส่วนของ Service Machine Learning
def predict(data : request_body):
    info = [data.height,data.weight,data.age,data.sleep,data.somtum]
    pred = predict_gender(info)
    return {f'{pred}'}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)