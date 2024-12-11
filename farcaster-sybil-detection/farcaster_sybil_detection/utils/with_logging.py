from functools import wraps
import logging

from typing import Protocol
from abc import ABC
from typing import Type


def add_logging(cls: Type):
    """Class decorator to add logging capabilities"""
    original_init = cls.__init__

    @wraps(cls.__init__)
    def new_init(instance, *args, **kwargs):
        # Setup logging
        instance.logger = logging.getLogger(cls.__name__)
        instance.logger.setLevel(logging.DEBUG)

        if not instance.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            instance.logger.addHandler(handler)

        # Call original init
        original_init(instance, *args, **kwargs)

    cls.__init__ = new_init
    return cls


class HasLogger(Protocol):
    """Protocol for objects with a logger"""

    logger: logging.Logger


class LoggedBase:
    """Base class that provides logging functionality"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)


class LoggedABC(ABC, LoggedBase):
    """Abstract base class with logging support"""

    pass
