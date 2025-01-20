def f1(func):
    def warpper(x, y):
        if x<=0:
            raise ValueError("Input must be positive")
        return func(x+y)
    # def warpper(x):
    #     if x<=0:
    #         raise ValueError("Input must be positive")
    #     return func(x)

    return warpper
@f1
def f2(x):
    return x + 1

if __name__ == "__main__":
    v= f2(4,3)
    """
    执行过程：
    warpper(4,3)->f2(4+3)

    """
    print(v) # 8
