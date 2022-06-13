import time


def timed(func):
    def wrapper(*args, **kwargs):
        t_start = time.perf_counter()
        res = func(*args, **kwargs)
        elapsed = time.perf_counter() - t_start
        return elapsed, res
    return wrapper