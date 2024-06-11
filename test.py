import sys

def my_function(arg1, arg2):
    # 函数体
    print("参数 1:", arg1)
    print("参数 2:", arg2)

if __name__ == '__main__':
    # 获取命令行参数
    args = sys.argv[1:]  # 跳过第一个参数，它是可执行文件的名称

    # 检查参数数量
    if len(args) != 2:
        print("需要提供两个参数")
        sys.exit(1)

    # 调用函数
    my_function(args[0], args[1])