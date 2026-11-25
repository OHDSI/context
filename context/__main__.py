import simple_parsing

from context.args import Args
from context.train import train


def main():
    train(simple_parsing.parse(Args))


if __name__ == "__main__":
    main()
