"""Response schemas for private cocktail image uploads."""

from pydantic import BaseModel


class UploadedCocktailImage(BaseModel):
    cocktail_id: int
    cocktail_name: str
    source_filename: str
    image_url: str
    thumbnail_url: str


class BatchImageUploadResponse(BaseModel):
    uploaded: int
    items: list[UploadedCocktailImage]
