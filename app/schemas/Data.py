from pydantic import BaseModel


class DataIn(BaseModel):
    no_of_dependents: int
    education: bool
    self_employed: bool
    income_annum: int
    loan_amount: int
    loan_term: int
    cibil_score: int
    total_assets: int