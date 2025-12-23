
def dotwice():

    def decorator(func):
        def wraper(*args, **kwargs):
            func(*args, **kwargs)  # 第一次执行
            func(*args, **kwargs)  # 第二次执行
        return wraper
    return decorator

#

@dotwice()
def add(a:int,b:int):
    print(a+b)
    return 0


add(2,4)