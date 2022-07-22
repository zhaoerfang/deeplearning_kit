import sys

from package import mod1

print(sys.path)

"""
如果在根目录直接通过 "python package2/run.py" 运行，此时sys.path的第一个路径是"./package2"，无法调用mod1；
而将脚本当作模块执行，即使用“python -m package2.run”运行，此时sys.path的第一个路径是根目录，可以搜索到“package”，因此可以调用mod1。
"""

if __name__ == "__main__":
    print(f"in run, name = {__name__}")

