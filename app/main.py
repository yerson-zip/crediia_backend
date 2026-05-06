from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.router import regresion, red_neuronal

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://crediia-front.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(regresion.router)
app.include_router(red_neuronal.router)
@app.get("/")
async def root():
    return {"message": "Holaaa, por ahora no hay configuración de CORS, no haga nada indebido"}

