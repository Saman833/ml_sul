from pydantic import BaseModel, UUID4, StrictStr

from config import START_MODULE_MESSAGE

MODULE_DESCRIPTION = "This module is used for Fast Api models (classes) storage"


class GetMessageRequestModel(BaseModel):
    dialog_id:       UUID4
    last_msg_text:   StrictStr
    last_message_id: UUID4 | None


class GetMessageResponseModel(BaseModel):
    new_msg_text: StrictStr
    dialog_id:    UUID4


def main():
    print(START_MODULE_MESSAGE + str(__file__))
    print(MODULE_DESCRIPTION)


if __name__ == "__main__":
    main()
