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

async def get_current_user_from_websocket_header( # Renamed function
    websocket: WebSocket,
    db = Depends(open_db_connection)
) -> Optional[UserInDB]:
    logger.debug(f"Attempting WebSocket authentication via headers. URL: {websocket.url}")
    
    token: Optional[str] = None
    auth_header_found = False
    # Headers in ASGI scope are list of (byte_string_key, byte_string_value)
    for name, value in websocket.scope.get("headers", []):
        if name.lower() == b"authorization":
            auth_header_found = True
            header_value_str = value.decode("utf-8")
            parts = header_value_str.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1]
                logger.debug(f"Extracted token from Authorization header: {token[:10]}...") # Log only prefix
            else:
                logger.warning(f"Malformed Authorization header found: {header_value_str}")
            break # Found Authorization header, no need to continue loop

    if not auth_header_found:
        logger.warning("WebSocket connection attempt without 'Authorization' header.")
        return None
    if not token:
        logger.warning("WebSocket 'Authorization' header present but no Bearer token found or malformed.")
        return None

    try:
        credentials_exception = HTTPException(
            status_code=status.WS_1008_POLICY_VIOLATION, # WebSocket specific status code
            detail="Could not validate credentials from Authorization header",
        )

        logger.debug("Calling verify_token_and_get_data...")
        token_data = verify_token_and_get_data(token, credentials_exception)

        if not token_data:
            logger.warning("verify_token_and_get_data returned None without raising exception.")
            return None

        logger.info(f"Token verified successfully for user: {token_data.username}")

        logger.debug(f"Attempting to fetch user '{token_data.username}' from DB...")
        user = crud.get_user_by_username(db, username=token_data.username)

        if user:
            logger.info(f"Successfully authenticated WebSocket user via header: {user.username} (ID: {user.id})")
            return user
        else:
            logger.error(f"User '{token_data.username}' found in token but not in DB.")
            return None

    except ExpiredSignatureError:
        logger.warning("WebSocket token validation failed (header auth): ExpiredSignatureError")
        return None
    except JWTError as e:
        logger.warning(f"WebSocket token validation failed (header auth): {type(e).__name__}: {e}")
        return None
    except HTTPException as e:
        logger.warning(f"WebSocket token validation failed due to HTTPException (header auth): {e.detail} (Status: {e.status_code})")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error during WebSocket authentication (header auth): {e}")
        return None