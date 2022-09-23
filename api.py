from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import joblib
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
import uvicorn
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from fastapi.middleware.cors import CORSMiddleware

class Nanocement(BaseModel):
    wc_ratio: float
    curing_time: float
    nanosilica: float
    microsilica: float

app = FastAPI(title = 'Model for predicting compressive strength of concerete from nanoparticles', 
            description = 'Accurately predicting compressive strength of concerete from nanoparticles!')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials = True,
    allow_methods =["*"],
    allow_headers = ["*"]
)

@app.get("/", response_class=PlainTextResponse)
def home():
    return "Welcome! API is working perfectly well. Use /docs to proceed to make to determing the compressive of concretes."


@app.post("/predict")
def make_prediction(nano: Nanocement):
    wc_ratio = nano.wc_ratio
    curing_time = nano.curing_time
    nanosilica = nano.nanosilica
    microsilica = nano.microsilica

    data = [[wc_ratio, curing_time, nanosilica, microsilica]]

    sc = StandardScaler()
    df = pd.read_excel("Nano.xlsx")
    df.drop(columns = ['No.'], inplace=True)
    df.drop(df[df['Curing time (Days)'] == 90 ].index, inplace = True)
    df.columns = ['wc_ratio', 'curing_time', 'nanosilica', 'microsilica', 'compressive_strength']
    X = df.drop('compressive_strength', axis=1)

    processed = sc.fit_transform(X)
   
    processed_data = sc.transform(data)

    loaded_model = joblib.load(open('Model.pkl', 'rb'))
    prediction = loaded_model.predict(processed_data)
    prediction = list(prediction.round(2))
    
    for i in prediction:
        output = i

        return {"strength": output}


#run
if __name__ == '__main__':
    uvicorn.run(app)
    