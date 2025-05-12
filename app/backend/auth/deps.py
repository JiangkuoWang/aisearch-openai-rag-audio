# deps.py
import logging # Added
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, ExpiredSignatureError # Removed InvalidSignatureError import
from typing import Optional, Annotated # Added Annotated
from fastapi import WebSocket # Added WebSocket
from .security import verify_token_and_get_data
from . import crud
from .db import open_db_connection
from .schemas import UserInDB # Corrected import source for type hinting

# Added logger
logger = logging.getLogger("voicerag.auth.deps") # Use a specific logger name

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login/token", auto_error=False) # auto_error=False for optional token

async def get_current_user(token: str = Depends(oauth2_scheme), db = Depends(open_db_connection)) -> UserInDB:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token: # Added check for missing token when auto_error=False
        raise credentials_exception

    token_data = verify_token_and_get_data(token, credentials_exception)
    if not token_data: # Should not happen if verify_token_and_get_data raises, but as a safeguard
        raise credentials_exception
        
    user = crud.get_user_by_username(db, username=token_data.username)
    
    if user is None:
        raise credentials_exception
    return user

async def get_current_user_or_none(token: Optional[str] = Depends(oauth2_scheme), db = Depends(open_db_connection)) -> Optional[UserInDB]:
    if not token:
        return None
    try:
        # Define a dummy credentials_exception for the call, even though we'll catch exceptions
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials (optional auth)",
            headers={"WWW-Authenticate": "Bearer"},
        )
        token_data = verify_token_and_get_data(token, credentials_exception) # Pass the required argument
        if not token_data: # If token is invalid (e.g. expired, malformed)
            logger.debug("verify_token_and_get_data returned None in optional auth path.")
            return None
        user = crud.get_user_by_username(db, username=token_data.username)
        return user # Returns user or None if not found
    except JWTError as e: # Catch JWT errors specifically
        logger.debug(f"JWTError during optional auth: {e}")
        return None
    except HTTPException as e: # Catch other HTTPExceptions that verify_token_and_get_data might raise if not handled
        logger.debug(f"HTTPException during optional auth: {e.detail}")
        return None

async def get_current_user_from_websocket_query_param(
    websocket: WebSocket,
    # Removed token parameter from signature, will get it directly from websocket object
    db = Depends(open_db_connection)
) -> Optional[UserInDB]:
    logger.debug(f"Attempting WebSocket authentication. URL: {websocket.url}")
    # Get token directly from query parameters inside the function
    token = websocket.query_params.get("token")

    if not token:
        logger.warning("WebSocket connection attempt without 'token' query parameter.")
        return None

    logger.debug(f"Extracted token from query param: {token[:10]}...") # Log only prefix for security

    try:
        # Define credentials_exception inside try block as it's only needed on failure path
        credentials_exception = HTTPException(
            status_code=status.WS_1008_POLICY_VIOLATION, # WebSocket specific status code
            detail="Could not validate credentials from query parameter",
        )

        logger.debug("Calling verify_token_and_get_data...")
        token_data = verify_token_and_get_data(token, credentials_exception)

        if not token_data:
            # This case might be less likely if verify_token_and_get_data raises exceptions properly
            logger.warning("verify_token_and_get_data returned None without raising exception.")
            return None

        logger.info(f"Token verified successfully for user: {token_data.username}")

        logger.debug(f"Attempting to fetch user '{token_data.username}' from DB...")
        user = crud.get_user_by_username(db, username=token_data.username)

        if user:
            logger.info(f"Successfully authenticated WebSocket user: {user.username} (ID: {user.id})")
            return user
        else:
            logger.error(f"User '{token_data.username}' found in token but not in DB.")
            return None

    except ExpiredSignatureError:
        logger.warning("WebSocket token validation failed: ExpiredSignatureError")
        return None
    # Removed specific catch for InvalidSignatureError
    except JWTError as e:
        # Log the specific type of JWTError encountered
        logger.warning(f"WebSocket token validation failed: {type(e).__name__}: {e}")
        return None
    except HTTPException as e: # If verify_token_and_get_data raises HTTP exception
        logger.warning(f"WebSocket token validation failed due to HTTPException: {e.detail} (Status: {e.status_code})")
        # The endpoint should handle this by closing the WebSocket.
        # Returning None means the endpoint will see it as unauthenticated.
        return None
    except Exception as e:
        logger.exception(f"Unexpected error during WebSocket authentication: {e}")
        return None