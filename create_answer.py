
from typing import List
from openai import OpenAI
from httpx import Client
from uuid import UUID

from config import START_MODULE_MESSAGE, Config, logger, str_object_is_created, PRE_PROMPT, DEEPSEEK_BASE_URL
from database import select_messages_by_dialog  # новый импорт

MODULE_DESCRIPTION = "This module is used for creating answer on /get_message"


def query_openai_proxy(prompt: str,
                 model: str = "deepseek-chat",
                 use_proxy = Config().USE_PROXY) -> str:
    """
        Do a request to OpenAI API with proxy
            Parameters:
                prompt (str): Prompt for the model,
                model (str): Model name
                use_proxy (bool): Use proxy or not
            Returns:
                dict: Ответ от OpenAI API.
    """

    logger.info("model: " + model)
    logger.info("prompt: {" + str(prompt) + "}")

    if use_proxy:
        logger.info("Using proxy: " + Config().PROXY_URL)
        client = OpenAI(api_key=Config().OPEN_AI_API_KEY, http_client=Client(proxy=Config().PROXY_URL),
                        base_url=DEEPSEEK_BASE_URL if Config().MODEL_OPERATOR == "deepseek" else None)
    else:
        client = OpenAI(api_key=Config().OPEN_AI_API_KEY,
                        base_url=DEEPSEEK_BASE_URL if Config().MODEL_OPERATOR == "deepseek" else None)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
    )

    logger.info(str(chat_completion))
    logger.info(chat_completion.choices[0].message.content)

    client.close()

    return chat_completion.choices[0].message.content


def check_proxy_status() -> bool:
    """
    Функция проверяет работоспособность прокси путем запроса к http://httpbin.org/ip
    Returns:
        bool: True если прокси работает, False если не работает
    """

    test_proxy_url = "http://httpbin.org/ip"
    logger.info(f"request to url: {test_proxy_url} to test proxy")

    try:
        with Client(proxy=Config().PROXY_URL, timeout=5) as test_client:
            response = test_client.get(test_proxy_url)
            logger.info(f"response: {response}")
            logger.info(f"response: {response.text}")
            if response.status_code == 200:
                logger.info("Прокси работает")
                return True
            else:
                logger.error(f"Прокси не работает: статус {response.status_code}")
                return False
    except Exception as e:
        logger.error(f"Прокси не работает: {str(e)}")
        return False


def get_response_from_llm(dialog_id: UUID, request_message_text: str) -> str:

    response_message_text: str = ""

    prompt: str = ""

    if dialog_id:
        try:
            logger.info(f"Получаем историю переписки по dialog_id: {dialog_id}")
            history_messages = select_messages_by_dialog(dialog_id)
            history_text = ""
            if history_messages:
                for msg in history_messages:
                    history_text += f"{msg['participant_index']}: {msg['text']}\n"

            # Объединяем preprompt, историю диалога
            prompt = PRE_PROMPT + "\n" + history_text
        except Exception as e:
            logger.error(f"Problems with database. Error while selecting messages: {str(e)}")
    else:
        logger.error("dialog_id is None. Not getting history messages")

    if prompt == "":
        logger.error("No history messages. Using only current")
        prompt = PRE_PROMPT + request_message_text

    proxy_is_working: bool = False

    if Config().USE_PROXY:
        proxy_is_working: bool = check_proxy_status()

    logger.info(f"proxy_is_working: {proxy_is_working}")

    model: str = get_model_name_for_model_operator(Config().MODEL_OPERATOR)

    try:
        response_message_text = query_openai_proxy(
            prompt=prompt,
            model=model,
            use_proxy=proxy_is_working
        )
    except Exception as e:
        logger.error(f"Error in query_openai_proxy: {str(e)}")

    if response_message_text == "":
        logger.error("No response from OpenAI API")
        logger.error("Returning echo response")
        response_message_text = request_message_text

    return response_message_text


def get_model_name_for_model_operator(model_operator: str) -> str:
    model_name:str

    default_model_name: str = "deepseek-chat"

    model_name: str = default_model_name
    if model_operator == "deepseek":
        model_name: str = "deepseek-chat"
    elif model_operator == "openai":
        model_name: str = "gpt-4o"
    else:
        logger.error(f"Unknown model_operator: {model_operator}. Using default model name: {default_model_name}")
        model_name: str = default_model_name

    return model_name


def main():
    print(START_MODULE_MESSAGE + str(__file__))
    print(MODULE_DESCRIPTION)


if __name__ == "__main__":
    main()
