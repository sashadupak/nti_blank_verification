from enum import Enum
class BlankStatus(Enum):
    CONFIRMED = "подписнный бланк"
    CORRECT = "нужный бланк, не подписан"
    OUTOFSHOT = "часть бланка не в кадре"
    WRONG = "не бланк"