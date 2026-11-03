import fire
import sys

from wuji_dl.runner import Runner, fit_runner, test_runner, export_onnx_runner


def main():
    sys.path.append(".")
    fire.Fire(Runner)


def fit():
    sys.path.append(".")
    fire.Fire(fit_runner)


def test():
    sys.path.append(".")
    fire.Fire(test_runner)

def export_onnx():
    sys.path.append(".")
    fire.Fire(export_onnx_runner)