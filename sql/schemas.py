from pydantic import BaseModel

class NewsBase(BaseModel):
    title: str
    summary: str
    content: str
    source: str
    link: str

class NewsCreate(NewsBase):
    pass

class News(NewsBase):
    id: int

    class Config:
        orm_mode = True