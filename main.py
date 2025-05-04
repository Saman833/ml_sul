import uvicorn
import psycopg2
from fastapi import FastAPI
from uuid import uuid4, UUID

import time
from config import START_MODULE_MESSAGE, logger, str_object_is_created, Config
from models_fast_api import GetMessageRequestModel, GetMessageResponseModel
from create_answer import get_response_from_llm
from database import insert_message, init_db


MODULE_DESCRIPTION = "Main module for starting program. FastAPI functions are here"


app = FastAPI()


@app.post("/get_message", response_model=GetMessageResponseModel)
async def get_message(body: GetMessageRequestModel):
    """
    This functions receives a message from HumanOrNot and returns a response
        Parameters (JSON from POST-request):
            body (GetMessageRequestModel): model with request data
                dialog_id (UUID4): ID of the dialog where the message was sent
                last_msg_text (str): text of the message
                last_message_id (UUID4): ID of this message

        Returns (JSON from response):
            GetMessageResponseModel: model with response data
                new_msg_text (str): Ответ бота
                dialog_id (str): ID диалога
    """

    response: GetMessageResponseModel

    dialog_id: UUID | None = body.dialog_id

    logger.info(f"body: {body}")
    try:
        insert_message(
            msg_id=body.last_message_id,
            dialog_id=body.dialog_id,
            text=body.last_msg_text,
            participant_index=0,
        )
    except Exception as e:
        logger.error(f"Problems with database. Error while inserting user message: {str(e)}")
        dialog_id:UUID | None = None

    response_from_llm: str = get_response_from_llm(dialog_id, body.last_msg_text)

    logger.info(f"response_from_llm: {response_from_llm}")

    try:
        insert_message(
            msg_id=uuid4(),
            dialog_id=body.dialog_id,
            text=response_from_llm,
            participant_index=1
        )
    except Exception as e:
        logger.error(f"Problems with database. Error while inserting gpt answer message: {str(e)}")

    response: GetMessageResponseModel = GetMessageResponseModel(
        new_msg_text=response_from_llm,
        dialog_id=body.dialog_id
    )

    logger.info(f"response: {response}")

    return response


@app.on_event("startup")
def on_startup() -> None:
    """
    Запуск приложения FastAPI.
    Выполняем проверку доступности PostgreSQL в цикле (на всякий случай)
    После успешного соединения инициализируем базу.
    """
    while True:
        try:
            conn = psycopg2.connect(
                database=Config().DB_NAME,
                user=Config().DB_USER,
                password=Config().DB_PASSWORD,
                host=Config().DB_HOST,
                port=Config().DB_PORT
            )
            conn.close()
            break
        except psycopg2.OperationalError:
            logger.warning("Waiting for PostgreSQL to become available...")
            time.sleep(2)

    # Инициализация БД
    init_db()



def main():
    logger.info(START_MODULE_MESSAGE + str(__file__))
    logger.info(MODULE_DESCRIPTION)
    logger.info(str_object_is_created(app))

    uvicorn.run(app, host="0.0.0.0", port=Config().PORT)


if __name__ == "__main__":
    main()
