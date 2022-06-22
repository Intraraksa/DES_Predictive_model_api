import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

class request_body(BaseModel):
    sex: str
    age: int
    os: str
    friend: str

model = pickle.load(open("socialtime_model.pickle", 'rb')) #โหลดโมเดลที่เรา Train ใน Colab
scaler = pickle.load(open("socialtime_scaler.pickle", 'rb')) #โหลด Scaler 
encoder = pickle.load(open("socialtime_encoder.pickle", 'rb')) #โหลด Encoder

def pred_pipeline(inp, model=model, sc=scaler, enc=encoder): # Machine learning prediction pipeline
    columns = ['sex', 'age', 'mobile os', 'have girl/boyfriend'] ## Column เรียงตามที่เรา Train
    names = enc.get_feature_names([columns[0]] + columns[2:]) 
    s = pd.DataFrame({c: [inp[i]] for i, c in enumerate(columns)})
    encoded = enc.transform(s[[columns[0]] + columns[2:]])
    encoded = pd.DataFrame(encoded.toarray(), columns=names)
    x_data = pd.concat([s['age'], encoded], axis=1)
    inp_norm = sc.transform(x_data)
    pred = model.predict(inp_norm) 
    return pred

@app.get('/')
def index():
    return {'message': 'This social time for prediction service'}

@app.post("/prediction")
def predict(data : request_body):
    info = [data.sex,data.age,data.os,data.friend]
    pred = pred_pipeline(info,model=model, sc=scaler, enc=encoder)
    return {"The social time is" : f'{pred}'}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)