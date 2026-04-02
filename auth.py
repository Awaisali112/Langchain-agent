import os
import sqlite3
import bcrypt
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from jose import jwt, JWTError
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET", "change-me-in-production-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

bearer_scheme = HTTPBearer()
router = APIRouter(prefix="/auth", tags=["auth"])

# ── DB setup ───────────────────────────────────────────────────────────────────
AUTH_DB = os.getenv("AUTH_DB", "auth.db")

def get_db():
    conn = sqlite3.connect(AUTH_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT    NOT NULL,
            email     TEXT    NOT NULL UNIQUE,
            password  TEXT    NOT NULL,
            created_at TEXT   DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ── Schemas ────────────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict

# ── Helpers ────────────────────────────────────────────────────────────────────
def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())

def create_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# ── Dependency: current user ───────────────────────────────────────────────────
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> dict:
    return decode_token(credentials.credentials)

# ── Routes ─────────────────────────────────────────────────────────────────────
@router.post("/register", response_model=AuthResponse)
def register(req: RegisterRequest):
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    conn = get_db()
    existing = conn.execute("SELECT id FROM users WHERE email = ?", (req.email,)).fetchone()
    if existing:
        conn.close()
        raise HTTPException(status_code=409, detail="Email already registered")

    hashed = hash_password(req.password)
    cursor = conn.execute(
        "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
        (req.name, req.email, hashed),
    )
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()

    token = create_token({"sub": str(user_id), "email": req.email, "name": req.name})
    return AuthResponse(
        access_token=token,
        user={"id": user_id, "name": req.name, "email": req.email},
    )


@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (req.email,)).fetchone()
    conn.close()

    if not row or not verify_password(req.password, row["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_token({"sub": str(row["id"]), "email": row["email"], "name": row["name"]})
    return AuthResponse(
        access_token=token,
        user={"id": row["id"], "name": row["name"], "email": row["email"]},
    )


@router.get("/me")
def me(current_user: dict = Depends(get_current_user)):
    return current_user
