import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, Query
from typing import List, Dict, Union
import logging
import os
import asyncio
import csv

# uvicorn main:app --reload

logging.basicConfig(level=logging.DEBUG)
app = FastAPI()


timeout_config = httpx.Timeout(
    connect=20.0,
    read=120.0,
    write=30.0,
    pool=60.0
)


@app.get("/deep_learning_api")
def deep_learning_model():
    pass




