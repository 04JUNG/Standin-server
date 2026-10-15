"""Standalone UI/BFF dev server; no rough inference or asset download on import."""
from fastapi import FastAPI
from .natural_search import router

app=FastAPI(title='Standin Natural Part Search Preview',version='1.0.0')
app.include_router(router)
