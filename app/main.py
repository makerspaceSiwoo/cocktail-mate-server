from fastapi import FastAPI

app = FastAPI()

mock_cocktails = [
    {
        "id": 1,
        "imageUrl": "https://fastly.picsum.photos/id/73/200/200.jpg",
        "name": "마가리타",
        "baseTag": "tequila",
        "description": "상큼하고 짭짤한 클래식 칵테일",
        "ABV": 21.3,
        "numLike": 10
    },
    {
        "id": 2,
        "imageUrl": "https://fastly.picsum.photos/id/74/200/200.jpg",
        "name": "모히또",
        "baseTag": "rum",
        "description": "민트와 라임 향이 강한 청량한 칵테일",
        "ABV": 13.0,
        "numLike": 25
    },
    {
        "id": 3,
        "imageUrl": "https://fastly.picsum.photos/id/75/200/200.jpg",
        "name": "진토닉",
        "baseTag": "gin",
        "description": "진과 토닉워터를 섞은 깔끔한 칵테일",
        "ABV": 12.5,
        "numLike": 18
    },
]

@app.get("/")
def root():
    return {"message": "Hello World"}



@app.get("/cocktail/{id}")
def root():
    return {"cocktailId": 123,
            "imageUrl" : "https://fastly.picsum.photos/id/73/200/200.jpg?hmac=IYjgRq-Ok9gn3_MVxJ4TlfhLPONQ97qWvp2Ir1Y1z6c",
            "cocktailName": "마가리타"
            }
    

@app.get("/search")
def search_cocktails(keyword: str = "", page: int = 1, rpp: int = 10):
    filtered = [
        cocktail for cocktail in mock_cocktails
        if keyword.lower() in cocktail["name"].lower()
        or keyword.lower() in cocktail["baseTag"].lower()
        or keyword.lower() in cocktail["description"].lower()
    ]

    total = len(filtered)

    start = (page - 1) * rpp
    end = start + rpp
    paged_items = filtered[start:end]

    return {
        "total": total,
        "cocktails": paged_items
    }

@app.get("/list")
def cockailLists():
    return mock_cocktails
# 칵테일 id

# 쿠키 - userId

# 응답

# 칵테일 정보 {
# id, 이미지, 이름, 베이스 태그(무알콜,럼,진,…), 설명,
# 도수, 좋아요 수 , 유저 좋아요 여부, 레시피
# }