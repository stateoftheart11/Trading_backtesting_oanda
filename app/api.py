from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from json.decoder import JSONDecodeError
from algorithm.backtesting import Backtesting
from starlette.responses import JSONResponse
import numpy as np

app = FastAPI()

load_dotenv()
api_key = os.environ.get('API_KEY')
backtesting = Backtesting(api_key)

ALLOWED_HOSTS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

def clean_data(data):
    if isinstance(data, dict):
        return {k: clean_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_data(v) for v in data]
    elif np.isnan(data) or np.isinf(data):
        return 0  # or None, or a custom value
    else:
        return data

@app.get("/", tags=["root"])
async def read_root(req: Request) -> dict:
    print(req.client)
    return {"message": "Welcome to your todo list."}

@app.post("/backtesting/")
async def post_backtesting(req: Request) -> dict:
    print("accept request backtesting !!!")
    try:
        request_backtesting = await req.json()
        print(request_backtesting)
        results = backtesting.run_backtest(request_backtesting["instrument"], request_backtesting["start_date"], request_backtesting["end_date"], request_backtesting["timeframe"], request_backtesting["input_parameters"], request_backtesting["trading_hours"])
        print(results)
        return {'result': clean_data(results)}

    except JSONDecodeError:
        # Do something when an invalid JSON string is encountered
        # For example, return an error message as a response
        return JSONResponse(status_code=400, content={"detail": "Invalid JSON payload"})