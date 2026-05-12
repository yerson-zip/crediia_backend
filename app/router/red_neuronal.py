from fastapi import  UploadFile, HTTPException, APIRouter
from app.schemas.Data import DataIn

from app.service.red_neuronal import processing_data, processing_data_file, evaluate_file

router = APIRouter(tags=["Red neuronal"], prefix="/NN")


@router.post("/unique")
def predict_regresion(data:DataIn):
    response = processing_data(data.model_dump())

    return response

@router.post("/path")
def predict_regresion_path(file:UploadFile):
    response = processing_data_file(file)

    if type(response)=="dict":
        raise HTTPException(status_code=404, detail=response["Message"])

    return response

@router.post("/evaluate")
async  def evaluate(file:UploadFile ):
    return evaluate_file(file)