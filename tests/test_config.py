import config

def test_default_config():
    assert config.APP_TITLE == "Memory-Augmented RAG"
    assert config.APP_ICON == "🧠"
    assert config.DEFAULT_USER_ID == "default_user"
    assert isinstance(config.CHUNK_SIZE, int)
    assert isinstance(config.TEMPERATURE, float)
