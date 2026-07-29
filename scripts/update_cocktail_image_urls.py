from app.core.database import SessionLocal
from cocktail_mate_db.models import Cocktail

IMAGE_BASE_URL = "https://images.cocktail-mate.com/cocktails"


def main():
    with SessionLocal() as db:
        cocktails = db.query(Cocktail).order_by(Cocktail.id).all()

        for cocktail in cocktails:
            image_url = f"{IMAGE_BASE_URL}/{cocktail.id}.webp"

            cocktail.image_url = image_url
            print(f"{cocktail.id}: {image_url}")

        db.commit()

    print("이미지 URL 업데이트 완료")


if __name__ == "__main__":
    main()
