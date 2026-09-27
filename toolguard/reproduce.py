"""One command: rebuild data, train, evaluate.  `python -m toolguard.reproduce`"""
import subprocess
import sys


def run(*args):
    print("+", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m", *args], check=True)


def main():
    run("toolguard.data")
    run("toolguard.eval_detector")
    run("toolguard.train")
    run("toolguard.train", "--train", "train_nohn", "--out", "models/toolguard-minilm-nohn")
    run("toolguard.evaluate")
    run("toolguard.val_sanity")


if __name__ == "__main__":
    main()
