# ---INFO-----------------------------------------------------------------------
"""
Settings for the project.
"""
# ---DEPENDENCIES---------------------------------------------------------------
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# ------------------------------------------------------------------------------
ENV_FILE = Path(__file__).parent.parent / "envs" / ".env"


# ---OpenAI Settings------------------------------------------------------------
class OpenAISettings(BaseSettings):
    """
    OpenAI API settings.
    """

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="OPENAI_",
        extra="ignore",
    )
    api_key: str
