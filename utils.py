import contextlib
import sys
from tqdm.contrib import DummyTqdmFile


@contextlib.contextmanager
def stream_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        print("## stream_redirect_tqdm-try")
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        print("## stream_redirect_tqdm-except")
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        print("## stream_redirect_tqdm-finally")
        sys.stdout, sys.stderr = orig_out_err
