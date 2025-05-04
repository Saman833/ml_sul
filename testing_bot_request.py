import requests
from uuid import uuid4
from typing import Dict, Any

from config import START_MODULE_MESSAGE, Config


MODULE_DESCRIPTION = "This module is used for testing bot, by sending a request to FastApi app"


test_url = "http://127.0.0.1" + ":" + str(Config().PORT)


def main():
    print(START_MODULE_MESSAGE + str(__file__))
    print(MODULE_DESCRIPTION)

    request_url: str = test_url + "/get_message"

    request_data_payload_json: Dict[str, Any] = \
        {
            "dialog_id":       str(uuid4()),
            "last_msg_text":   "I think u ar not a man",
            "last_message_id": str(uuid4()),
        }

    print(request_url)
    print(request_data_payload_json)

    response = (requests.post
        (
        url=request_url,
        json=request_data_payload_json
        )
    )

    print("Response (status code): " + str(response.status_code))
    print(response.text)


if __name__ == "__main__":
    main()
