from .LTSFRunner import LTSFRunner


def runner_select(name):
    name = name.upper()

    if name in ("LTSF", "LONG", "LONGTERM"):
        return LTSFRunner

    else:
        raise NotImplementedError
