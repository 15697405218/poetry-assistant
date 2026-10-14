from fastapi import APIRouter,Depends,HTTPException,status,Request,Response
from sqlalchemy.orm import Session
import uuid

from app.db import SessionLocal
from app.models.user import User
from app.schemas.auth import UserCreate, UserOut
from app.services.security import hash_password, verify_password
from app.services.redis_store import (
    blacklist_jti,
    create_session,
    drop_session,
    jti_blacklisted,
    limit_login_ip,
    login_fail,
    login_locked,
    login_ok,
    session_ttl,
    touch_session,
    ttl_until,
)

from datetime import datetime, timedelta, timezone

from fastapi.security import OAuth2PasswordRequestForm,OAuth2PasswordBearer
from jose import jwt,JWTError

from app.config import settings


router = APIRouter(prefix="/auth", tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
def create_access_token(sub: str, jti: str | None = None) -> tuple[str, str]:
    token_jti = jti or uuid.uuid4().hex
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_max_minutes
    )
    payload = {"sub": sub, "exp": expire, "jti": token_jti}
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, token_jti

@router.post("/login")
def login(
    request: Request,
    form:OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    ip = request.client.host if request and request.client else "unknown"
    limit_login_ip(ip)
    email = (form.username or "").strip()
    if login_locked(email):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="尝试次数过多，请稍后再试",
        )
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(form.password, user.hashed_password):
        login_fail(email)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="邮箱或者密码错误")
    login_ok(email)
    access_token, jti = create_access_token(str(user.id))
    create_session(jti)
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    exists = db.query(User).filter(User.email == payload.email).first()
    if exists:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="邮箱已被注册")

    user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def get_current_user(
    response: Response,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效的登录凭证",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        data = jwt.decode(
            token,
            settings.jwt_secret, 
            algorithms=[settings.jwt_algorithm])
        sub: str = data.get("sub")
        if sub is None:
            raise credentials_exception
        jti = data.get("jti")
        if jti and jti_blacklisted(str(jti)):
            raise credentials_exception
        if jti and not touch_session(str(jti)):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="登录已过期，请重新登录",
                headers={"WWW-Authenticate": "Bearer"},
            )
        user_id: int = int(sub)
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise credentials_exception
        if jti and ttl_until(data.get("exp")) < session_ttl():
            new_token, _ = create_access_token(sub, jti=str(jti))
            response.headers["X-Access-Token"] = new_token
        return user
    except JWTError:
        raise credentials_exception

@router.post("/logout")
def logout(token: str = Depends(oauth2_scheme)):
    try:
        data = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的登录凭证",
            headers={"WWW-Authenticate": "Bearer"},
        )
    jti = data.get("jti")
    if not jti:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="旧令牌没有 jti，等它过期即可",
        )
    drop_session(str(jti))
    blacklist_jti(str(jti), ttl_until(data.get("exp")))
    return {"ok": True}

@router.get("/me", response_model=UserOut)
def read_me(current_user: User = Depends(get_current_user)):
    return current_user