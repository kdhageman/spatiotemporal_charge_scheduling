import logging
import os
import pickle
import time


def timed(func):
    def wrapper(*args, **kwargs):
        t_start = time.perf_counter()
        res = func(*args, **kwargs)
        elapsed = time.perf_counter() - t_start
        return elapsed, res
    return wrapper

# file caching
def pickled(filepath):
    def f(func):
        def g(*args, **kwargs):
            try:
                if os.path.exists(filepath):
                    logging.debug("from file")
                with open(filepath, "'rb"):
                    res = pickle.load(filepath)
            except:
                logging.debug("from source")
                res = func(*args, **kwargs)
                logging.debug("to file")
                with open(filepath, 'wb'):
                    pickle.dump(res)
            return res
        return g
    return f