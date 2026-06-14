"""좋아요 비즈니스 로직. 응답은 기존 mock과 동일하게 유지."""
from app.like.repository import LikeRepository


class LikeService:
    def __init__(self, repository: LikeRepository | None = None) -> None:
        self.repository = repository or LikeRepository()

    def like_list(self) -> dict:
        return {
            "cocktails": [
                {
                    "cocktailId": 1,
                    "cocktailName": "마가리타",
                    "imageUrl": "https://fastly.picsum.photos/id/73/200/200.jpg",
                    "baseTag": "데킬라",
                    "likeCount": 128,
                    "isLiked": True,
                },
                {
                    "cocktailId": 2,
                    "cocktailName": "모히토",
                    "imageUrl": "https://fastly.picsum.photos/id/106/200/200.jpg",
                    "baseTag": "럼",
                    "likeCount": 95,
                    "isLiked": True,
                },
            ]
        }

    def like(self) -> dict:
        return {
            "cocktailId": 1,
            "isLiked": True,
            "likeCount": 129,
            "message": "좋아요 성공",
        }

    def unlike(self) -> dict:
        return {
            "cocktailId": 1,
            "isLiked": False,
            "likeCount": 128,
            "message": "좋아요 취소 성공",
        }
