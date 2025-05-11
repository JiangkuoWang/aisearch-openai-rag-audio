# deps.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from .security import verify_token_and_get_data
from . import crud
from .db import open_db_connection

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login/token")

async def get_current_user(token: str = Depends(oauth2_scheme), db = Depends(open_db_connection)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token_data = verify_token_and_get_data(token, credentials_exception)
    user = crud.get_user_by_username(db, username=token_data.username)
    
    if user is None:
        raise credentials_exception
    return user