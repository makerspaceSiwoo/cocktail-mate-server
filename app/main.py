from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    return {"cocktailId": id,
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
@app.get("/explore/{id}")
def root():
    return{
            "cocktailId": id,
            "cocktailName": "cocktailtail",
            "imageUrl": "https://fastly.picsum.photos/id/73/200/200.jpg?hmac=IYjgRq-Ok9gn3_MVxJ4TlfhLPONQ97qWvp2Ir1Y1z6c",
            "glass": "glass1",
            "ABV": 21.3,
            "numLike": 10,
            "recipe": "mix",
            "description": "salty, sugary",
            "baseTag": "rum",
            "isLiked": False
    }

@app.post("/signup")
def signup():
    return {
        "userId": 1,
        "nickname": "mockUser",
        "email": "mock@example.com",
        "message": "회원가입 성공"
    }


@app.post("/login")
def login():
    return {
        "userId": 1,
        "nickname": "mockUser",
        "email": "mock@example.com",
        "accessToken": "mock-access-token",
        "message": "로그인 성공"
    }


@app.post("/logout")
def logout():
    return {
        "message": "로그아웃 성공",
        "profileImageUrl": "https://fastly.picsum.photos/id/64/200/200.jpg"
    }


@app.get("/my/info")
def get_my_info():
    return {
        "userId": 1,
        "nickname": "mockUser",
        "email": "mock@example.com",
        "profileImageUrl": "https://fastly.picsum.photos/id/64/200/200.jpg"
    }


@app.get("/like/list")
def get_like_list():
    return {
        "cocktails": [
            {
                "cocktailId": 1,
                "cocktailName": "마가리타",
                "imageUrl": "https://fastly.picsum.photos/id/73/200/200.jpg",
                "baseTag": "데킬라",
                "likeCount": 128,
                "isLiked": True
            },
            {
                "cocktailId": 2,
                "cocktailName": "모히토",
                "imageUrl": "https://fastly.picsum.photos/id/106/200/200.jpg",
                "baseTag": "럼",
                "likeCount": 95,
                "isLiked": True
            }
        ]
    }


@app.post("/like")
def like_cocktail():
    return {
        "cocktailId": 1,
        "isLiked": True,
        "likeCount": 129,
        "message": "좋아요 성공"
    }


@app.delete("/unlike")
def unlike_cocktail():
    return {
        "cocktailId": 1,
        "isLiked": False,
        "likeCount": 128,
        "message": "좋아요 취소 성공"
    }


@app.get("/drink-of-the-day")
def root():
    return {
        "cocktailId": 123,
        "cocktailName": "magarita",
        "imageUrl":"https://fastly.picsum.photos/id/73/200/200.jpg?hmac=IYjgRq-Ok9gn3_MVxJ4TlfhLPONQ97qWvp2Ir1Y1z6c", 
        "description": "salty, sugary",
        "ABV": 21.3,
        }
# 칵테일 id

# 쿠키 - userId

# 응답

# 칵테일 정보 {
# id, 이미지, 이름, 베이스 태그(무알콜,럼,진,…), 설명,
# 도수, 좋아요 수 , 유저 좋아요 여부, 레시피
# }
