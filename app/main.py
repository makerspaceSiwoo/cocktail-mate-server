from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class SignupRequest(BaseModel):
    email: str
    password: str
    nickname: str


class LoginRequest(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    userId: int
    nickname: str
    email: str
    profileImageUrl: str | None = None


class SignupResponse(BaseModel):
    user: UserResponse
    message: str


class LoginResponse(BaseModel):
    user: UserResponse
    accessToken: str
    message: str


class MessageResponse(BaseModel):
    message: str


class CocktailLikeItemResponse(BaseModel):
    cocktailId: int
    cocktailName: str
    imageUrl: str
    baseTag: str
    likeCount: int
    isLiked: bool


class LikeListResponse(BaseModel):
    cocktails: list[CocktailLikeItemResponse]


class LikeResponse(BaseModel):
    cocktailId: int
    isLiked: bool
    likeCount: int
    message: str


class CocktailSimpleResponse(BaseModel):
    cocktailId: int
    imageUrl: str
    cocktailName: str


class CocktailListItemResponse(BaseModel):
    id: int
    imageUrl: str
    name: str
    baseTag: str
    description: str
    ABV: float
    numLike: int


class SearchCocktailResponse(BaseModel):
    total: int
    cocktails: list[CocktailListItemResponse]


class ExploreCocktailResponse(BaseModel):
    cocktailId: int
    cocktailName: str
    imageUrl: str
    glass: str
    ABV: float
    numLike: int
    recipe: str
    description: str
    baseTag: str
    isLiked: bool


class DrinkOfTheDayResponse(BaseModel):
    cocktailId: int
    cocktailName: str
    imageUrl: str
    description: str
    ABV: float


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


@app.get("/cocktail/{id}", response_model=CocktailSimpleResponse)
def get_cocktail(id: int):
    return {"cocktailId": id,
            "imageUrl" : "https://fastly.picsum.photos/id/73/200/200.jpg?hmac=IYjgRq-Ok9gn3_MVxJ4TlfhLPONQ97qWvp2Ir1Y1z6c",
            "cocktailName": "마가리타"
            }
    

@app.get("/search", response_model=SearchCocktailResponse)
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


@app.get("/list", response_model=list[CocktailSimpleResponse])
def cocktail_lists():
    return [
        {
            "cocktailId": cocktail["id"],
            "imageUrl": cocktail["imageUrl"],
            "cocktailName": cocktail["name"]
        }
        for cocktail in mock_cocktails
    ]


@app.get("/explore/{id}", response_model=ExploreCocktailResponse)
def get_explore_cocktail(id: int):
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

@app.post("/signup", response_model=SignupResponse)
def signup(request: SignupRequest):
    return {
        "user": {
            "userId": 1,
            "nickname": request.nickname,
            "email": request.email,
            "profileImageUrl": None
        },
        "message": "회원가입 성공"
    }


@app.post("/login", response_model=LoginResponse)
def login(request: LoginRequest):
    return {
        "user": {
            "userId": 1,
            "nickname": "mockUser",
            "email": request.email,
            "profileImageUrl": "https://fastly.picsum.photos/id/64/200/200.jpg"
        },
        "accessToken": "mock-access-token",
        "message": "로그인 성공"
    }


@app.post("/logout", response_model=MessageResponse)
def logout():
    return {
        "message": "로그아웃 성공"
    }


@app.get("/my/info", response_model=UserResponse)
def get_my_info():
    return {
        "userId": 1,
        "nickname": "mockUser",
        "email": "mock@example.com",
        "profileImageUrl": "https://fastly.picsum.photos/id/64/200/200.jpg"
    }


@app.get("/like/list", response_model=LikeListResponse)
def get_like_list():
    return {
        "cocktails": [
            {
                "cocktailId": 1,
                "cocktailName": "마가리타",
                "imageUrl": "https://fastly.picsum.photos/id/73/200/200.jpg",
                "baseTag": "tequila",
                "likeCount": 128,
                "isLiked": True
            },
            {
                "cocktailId": 2,
                "cocktailName": "모히또",
                "imageUrl": "https://fastly.picsum.photos/id/106/200/200.jpg",
                "baseTag": "rum",
                "likeCount": 95,
                "isLiked": True
            }
        ]
    }


@app.post("/cocktails/{cocktail_id}/like", response_model=LikeResponse)
def like_cocktail(cocktail_id: int):
    return {
        "cocktailId": cocktail_id,
        "isLiked": True,
        "likeCount": 129,
        "message": "좋아요 성공"
    }


@app.delete("/cocktails/{cocktail_id}/like", response_model=LikeResponse)
def unlike_cocktail(cocktail_id: int):
    return {
        "cocktailId": cocktail_id,
        "isLiked": False,
        "likeCount": 128,
        "message": "좋아요 취소 성공"
    }


@app.get("/drink-of-the-day", response_model=DrinkOfTheDayResponse)
def get_drink_of_the_day():
    return {
        "cocktailId": 123,
        "cocktailName": "magarita",
        "imageUrl":"https://fastly.picsum.photos/id/73/200/200.jpg?hmac=IYjgRq-Ok9gn3_MVxJ4TlfhLPONQ97qWvp2Ir1Y1z6c", 
        "description": "salty, sugary",
        "ABV": 21.3,
        }
