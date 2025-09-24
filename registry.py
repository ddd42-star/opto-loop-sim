# registry class
from inspect import signature

DEVICE_REGISTRY = {}


def device_registry(name):
    def decorator(func):
        DEVICE_REGISTRY[name] = func
        return func
    return decorator

