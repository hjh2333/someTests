from typing import Optional,Union


def fun(a, b: Union[int, str, None] = None, len: Optional[int] = None):
    print(a, b, len)

if __name__ == "__main__":
    fun(1)