import json
import os
import redis
from dotenv import load_dotenv

load_dotenv()

_client = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379"),
    decode_responses=True
)
SESSION_TTL = int(os.getenv("SESSION_TTL", 86400))  # 默认 24 小时
MAX_HISTORY_TURNS = 10

def get_history(session_id: str) -> list:
    try:
        data = _client.get(f"session:{session_id}")
        return json.loads(data) if data else []
    except redis.RedisError:
        return []

def set_history(session_id: str, history: list):
    try:
        if len(history) > MAX_HISTORY_TURNS * 2:
            history = history[-(MAX_HISTORY_TURNS * 2):]
        _client.setex(
            f"session:{session_id}",
            SESSION_TTL,
            json.dumps(history, ensure_ascii=False)
        )
    except redis.RedisError:
        pass
