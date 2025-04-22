# 创建一个简单的脚本来检查 LlamaIndex 中的类
import sys
import inspect

print("Checking LlamaIndex classes...")

try:
    # 尝试导入 llama_index.core.graph_stores.simple_labelled 模块
    from llama_index.core.graph_stores import simple_labelled
    
    # 打印模块中的所有对象
    print("\nAll names in simple_labelled module:")
    for name in dir(simple_labelled):
        # 不打印内部名称
        if not name.startswith('_'):
            obj = getattr(simple_labelled, name)
            # 检查是否是类
            if inspect.isclass(obj):
                print(f"Class: {name}")
            elif inspect.isfunction(obj):
                print(f"Function: {name}")
            else:
                print(f"Other: {name}")
    
    # 尝试导入整个 graph_stores 包
    print("\nChecking all graph store classes in llama_index.core.graph_stores:")
    from llama_index.core import graph_stores
    
    # 打印所有可用的 graph_stores 模块
    print("\nAll modules in graph_stores package:")
    for name in dir(graph_stores):
        if not name.startswith('_'):
            print(name)
            
    # 打印 SimpleGraphStore 的信息
    print("\nInformation about SimpleGraphStore:")
    from llama_index.core.graph_stores import SimpleGraphStore
    print(f"SimpleGraphStore class: {SimpleGraphStore}")
    print(f"SimpleGraphStore module: {SimpleGraphStore.__module__}")
    
    # 列出所有包含 "graph" 的类
    print("\nAll classes containing 'graph' in their name:")
    for module_name in dir(graph_stores):
        if not module_name.startswith('_'):
            try:
                module = getattr(graph_stores, module_name)
                for name in dir(module):
                    if 'graph' in name.lower() and not name.startswith('_'):
                        try:
                            obj = getattr(module, name)
                            if inspect.isclass(obj):
                                print(f"Class: {name} (in {module.__name__})")
                        except:
                            pass
            except:
                pass
    
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Error: {e}") 