import fire
import sys
# sys.path.append(r"C:\Users\t-jiahuihe\Code\PythonCode\TestForFirst")
from nnForward import testForward
def printHello():
    print("hello!")
    print("hhh")

def run(*args):
    arg_list = list(args)
    return '||'.join(arg_list)

def cli_main():
    fire.Fire(run)

if __name__ == '__main__':
    cli_main()