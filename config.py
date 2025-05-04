from loguru import logger
from typing import Any
from pydantic_settings import BaseSettings
from pydantic import computed_field
import os


# Python module info
START_MODULE_MESSAGE  = "You have launched the module "
WORKING_MODULE_MESSAGE = "Now this module will do its job"
MODULE_DESCRIPTION = "This module is used to save all creds and settings for the program"


# Logger settings
logger.add(
    "logs/app_{time}.log",
    rotation="00:00",
    retention="30 days",
    compression="zip",
)


# Creds from .env
class Config(BaseSettings):
    MODEL_OPERATOR: str = "deepseek"
    OPEN_AI_API_KEY: str
    PROXY_URL: str
    PORT: int = 8672
    USE_PROXY: bool = False

    DB_USER: str     = "student"
    DB_PASSWORD: str = "student_pass"
    DB_HOST: str     = "postgres"
    DB_PORT: int     = 5432
    DB_NAME: str     = "chat_db"

    @computed_field
    @property
    def db_url(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    class Config:
        env_file = ".env"


# Program settings
PRE_PROMPT_DEFAULT: str = ("Hello! You are participating in a Turing test. "
                           "Your task is to respond to the person's phrase "
                           "in the most human-like and convincing way. Use "
                           "the same language as the person did. If the "
                           "language is unknown to you, explain in a natural"
                           " and polite manner that you don't understand it."
                           " Your goal is to deceive the person into"
                           " thinking you are human. Now continue the"
                           " conversation with your next response."
                           "Here is the conversation so far:\n")


DEEPSEEK_BASE_URL = "https://api.deepseek.com"


preprompt_file = os.path.join(os.path.dirname(__file__), "preprompt.txt")
try:
    with open(preprompt_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
    PRE_PROMPT: str = content if content else PRE_PROMPT_DEFAULT
except FileNotFoundError:
    PRE_PROMPT: str = PRE_PROMPT_DEFAULT


# Function for dynamic logger information (Python module info)
def str_object_is_created(created_object: Any) -> str:
    """
        Function for making string with description of creation of object
            Parameters:
                created_object: creation of this object we should describe
            Returns:
                str: description of creation of object
    """
    return f"Object {str(created_object)} is created"


def main():
    print(START_MODULE_MESSAGE + str(__file__))
    print(MODULE_DESCRIPTION)


if __name__ == "__main__":
    main()

