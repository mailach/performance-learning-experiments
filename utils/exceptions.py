import logging


def handle_exception(error_message: str):
    def wrap(f):
        def wrapped_f(*args):
            try:
                return f(*args)
            except Exception as e:
                logging.error(
                    f"Error occoured in function {f.__name__}: {error_message}"
                )
                raise e

        return wrapped_f

    return wrap
