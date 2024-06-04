from datetime import timedelta
import time


def timer(function):
    def wrapper(*args, **kwargs):
        start = time.time()
        function(*args, **kwargs)
        end = time.time()
        time_elapsed = round(end - start, 2)
        time_elapsed = timedelta(seconds=time_elapsed)
        print(f"Training finished in {time_elapsed} (hh:mm:ss).")

    return wrapper
