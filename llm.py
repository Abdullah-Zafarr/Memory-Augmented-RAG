import config, logging
from groq import Groq
logger = logging.getLogger(__name__)

_c = None
def _get():
    global _c
    if not _c:
        if not config.GROQ_API_KEY: raise EnvironmentError("No API Key")
        _c = Groq(api_key=config.GROQ_API_KEY)
    return _c

def generate(msgs, model=None, tokens=None, temp=None):
    res = _get().chat.completions.create(
        model=model or config.GROQ_MODEL,
        messages=msgs,
        max_tokens=tokens or config.MAX_TOKENS,
        temperature=temp if temp is not None else config.TEMPERATURE
    )
    return res.choices[0].message.content or ""

def build_messages(sys, user):
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]
