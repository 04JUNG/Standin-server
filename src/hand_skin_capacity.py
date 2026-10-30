"""Dependency-free limits shared by runtime configuration and deployment tools."""

MAX_SCREEN_CONCURRENCY = 5


def validate_concurrency(value):
    if type(value) is not int or not 1 <= value <= MAX_SCREEN_CONCURRENCY:
        raise ValueError('hand screen concurrency must be an integer in [1,5]')
    return value
