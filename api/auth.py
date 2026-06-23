"""
Authentication Module — SecOpsAI API
JWT-based authentication for all detection endpoints.
Uses sha256_crypt instead of bcrypt to avoid the 72-byte limit.
"""

import os
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer


SECRET_KEY           = os.getenv("SECRET_KEY", "secopsai-dev-secret-change-in-production")
ALGORITHM            = "HS256"
TOKEN_EXPIRE_MINUTES = 60

# sha256_crypt has no byte length restriction unlike bcrypt
pwd_context   = CryptContext(schemes=["sha256_crypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

USERS_DB = {
    "analyst": {
        "username":        "analyst",
        "hashed_password": pwd_context.hash("secopsai123"),
        "role":            "analyst"
    },
    "admin": {
        "username":        "admin",
        "hashed_password": pwd_context.hash("admin123"),
        "role":            "admin"
    }
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(username: str, password: str):
    user = USERS_DB.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire    = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload  = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = USERS_DB.get(username)
    if user is None:
        raise credentials_exception
    return user
