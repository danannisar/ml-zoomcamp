import bentoml

from bentoml.io import JSON

from pydantic import BaseModel

class CreditApplication(BaseModel):
    seniority: int
    home: str
    time: int
    age: int
    marital: str
    records: str
    job: str
    expenses: int
    income: float
    assets: float
    debt: float
    amount: int
    price: int

model_ref = bentoml.xgboost.get("credit_risk_model:aiohjwcuoszjqaav")
dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service("credit_risk_classifier", runners=[model_runner])

# Define an endpoint on the BentoML service
# pass pydantic class application
@svc.api(input=JSON(pydantic_model=CreditApplication), output=JSON()) # decorate endpoint as in json format for input and output
async def classify(credit_application): # parallelized requests at endpoint level (async)
    # transform pydantic class to dict to extract key-value pairs 
    application = credit_application.dict()
    # transform data from client using dictvectorizer
    vector = dv.transform(application)
    # make predictions using 'runner.predict.run(input)' instead of 'model.predict'
    prediction = await model_runner.predict.async_run(vector) 
    # bentoml inference level parallelization (async_run)
    print(prediction)
    
    result = prediction[0]
    if result > 0.5 :
        return { "status": "DECLINED" }
    elif result > 0.25 :
        return { "status": "MAYBE" }
    else:
        return { "status": "APPROVED" }

