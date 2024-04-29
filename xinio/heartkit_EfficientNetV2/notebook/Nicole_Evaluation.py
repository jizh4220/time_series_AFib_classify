from heartkit.defines import (
    HKDemoParams, HeartBeat, HeartRate, HeartRhythm, HeartSegment, HKTestParams
)
from pydantic import BaseModel
from heartkit.tasks import TaskFactory
from typing import Type, TypeVar, List, Dict
from argdantic import ArgField, ArgParser

import os

cli = ArgParser()
B = TypeVar("B", bound=BaseModel)

def parse_content(cls: Type[B], content: str) -> B:
    """Parse file or raw content into Pydantic model.

    Args:
        cls (B): Pydantic model subclasss
        content (str): File path or raw content

    Returns:
        B: Pydantic model subclass instance
    """
    if os.path.isfile(content):
        with open(content, "r", encoding="utf-8") as f:
            content = f.read()

    return cls.model_validate_json(json_data=content)




task="arrhythmia"
task_handler = TaskFactory.get(task)

# In demo we will cover 5 regions at a time, frame_size*5
config = 'configs/AFIB_Ident-100class-2_Official.json'

task_handler.evaluate(parse_content(HKTestParams, config))