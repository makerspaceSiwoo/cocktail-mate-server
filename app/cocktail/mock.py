"""칵테일 mock 데이터.

ERD/DB 연동 전까지 repository가 이 데이터를 반환한다. 응답 값은 기존 구현과 동일하게 유지.
"""

MOCK_COCKTAILS = [
    {
        "id": 1,
        "imageUrl": "https://fastly.picsum.photos/id/73/200/200.jpg",
        "name": "마가리타",
        "baseTag": "tequila",
        "description": "상큼하고 짭짤한 클래식 칵테일",
        "ABV": 21.3,
        "numLike": 10,
    },
    {
        "id": 2,
        "imageUrl": "https://fastly.picsum.photos/id/74/200/200.jpg",
        "name": "모히또",
        "baseTag": "rum",
        "description": "민트와 라임 향이 강한 청량한 칵테일",
        "ABV": 13.0,
        "numLike": 25,
    },
    {
        "id": 3,
        "imageUrl": "https://fastly.picsum.photos/id/75/200/200.jpg",
        "name": "진토닉",
        "baseTag": "gin",
        "description": "진과 토닉워터를 섞은 깔끔한 칵테일",
        "ABV": 12.5,
        "numLike": 18,
    },
]
