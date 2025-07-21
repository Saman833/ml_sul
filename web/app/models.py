from pydantic import BaseModel, UUID4, StrictStr


class GetMessageRequestModel(BaseModel):
    dialog_id: UUID4
    last_msg_text: StrictStr
    last_message_id: UUID4 | None 