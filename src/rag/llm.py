from src.core import config
import logging
import time
from groq import Groq
from src.core.exceptions import LLMServiceError

logger = logging.getLogger(__name__)

_c = None
def _get():
    global _c
    if not _c:
        if not config.GROQ_API_KEY:
            raise EnvironmentError("No API Key configured for Groq LLM")
        _c = Groq(api_key=config.GROQ_API_KEY)
    return _c

def generate(msgs, model=None, tokens=None, temp=None, max_retries=3):
    client = _get()
    for attempt in range(max_retries):
        try:
            res = client.chat.completions.create(
                model=model or config.GROQ_MODEL,
                messages=msgs,
                max_tokens=tokens or config.MAX_TOKENS,
                temperature=temp if temp is not None else config.TEMPERATURE
            )
            if not res.choices:
                return ""
            return res.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"Groq API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise LLMServiceError(f"Failed to generate response after {max_retries} attempts: {e}")
            time.sleep(1 * (attempt + 1))
    return ""

def build_messages(sys, user):
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]
