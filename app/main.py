from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Hello World"}



@app.get("/cocktail/{id}")
def root():
    return {"cocktailId": 123,
            "imageUrl" : "https://fastly.picsum.photos/id/73/200/200.jpg?hmac=IYjgRq-Ok9gn3_MVxJ4TlfhLPONQ97qWvp2Ir1Y1z6c",
            "cocktailName": "마가리타"
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
        "message": "로그아웃 성공"
    }


# 칵테일 id

# 쿠키 - userId

# 응답

# 칵테일 정보 {
# id, 이미지, 이름, 베이스 태그(무알콜,럼,진,…), 설명,
# 도수, 좋아요 수 , 유저 좋아요 여부, 레시피
# }
