# app/backend/scripts/create_llama_graph_index.py
import locale
import os
import sys
import codecs

# --- Start: Force UTF-8 Globally (Place at the very top) ---
ENCODING = "utf-8"

def force_utf8():
    # print(f"Attempting to force Python default encoding to {ENCODING}...")
    try:
        # 1. Set locale environment variables
        os.environ['LANG'] = f'en_US.{ENCODING}'
        os.environ['LC_ALL'] = f'en_US.{ENCODING}'
        print(f"Set LANG/LC_ALL environment variables to en_US.{ENCODING}")
    except Exception as e:
        print(f"Warning: Could not set LANG/LC_ALL environment variables: {e}")

    try:
        # 2. Set Python's locale
        locale.setlocale(locale.LC_ALL, f'en_US.{ENCODING}')
        print(f"Successfully set locale.LC_ALL to en_US.{ENCODING}")
    except locale.Error:
        print(f"Warning: Locale 'en_US.{ENCODING}' not supported. Trying alternatives.")
        try:
            locale.setlocale(locale.LC_ALL, '') # Use system default locale first
            print(f"Set locale.LC_ALL to system default: {locale.getlocale()}")
            # Check if the system default is already UTF-8
            if codecs.lookup(locale.getpreferredencoding()).name != ENCODING:
                 print(f"Warning: System default encoding is not {ENCODING}.")
                 # Attempt platform-specific UTF-8 locale
                 utf8_locale = "" # TODO: Add specific logic for windows/linux if needed
                 if sys.platform == 'win32':
                      # Potential Windows UTF-8 locales (may vary)
                     locales_to_try = ['.UTF-8', '.65001'] # 65001 is UTF-8 code page
                 else:
                     # Common Linux/macOS UTF-8 locales
                     locales_to_try = ['C.UTF-8', 'en_US.UTF-8']
                 for loc in locales_to_try:
                     try:
                         locale.setlocale(locale.LC_ALL, loc)
                         print(f"Successfully set locale.LC_ALL to {loc}")
                         utf8_locale = loc
                         break # Success
                     except locale.Error:
                         continue # Try next locale
                 if not utf8_locale:
                     print("Error: Could not find/set a supported UTF-8 locale.")
        except Exception as e:
            print(f"Error setting locale: {e}")

    try:
        # 3. Set PYTHONIOENCODING environment variable
        os.environ["PYTHONIOENCODING"] = ENCODING
        print(f"Set PYTHONIOENCODING environment variable to {ENCODING}")
    except Exception as e:
        print(f"Warning: Could not set PYTHONIOENCODING: {e}")

    try:
        # 4. Reconfigure stdout/stderr (force UTF-8)
        # Use 'errors=replace' to avoid crashing on unencodable chars during print
        sys.stdout = codecs.getwriter(ENCODING)(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter(ENCODING)(sys.stderr.buffer, 'replace')
        print(f"Reconfigured sys.stdout and sys.stderr to use {ENCODING}")
    except Exception as e:
        print(f"Warning: Failed to reconfigure stdout/stderr: {e}")

    # Verify effective encoding - REMOVED problematic lines
    # print(f"sys.stdout.encoding: {sys.stdout.encoding}")
    # print(f"sys.stderr.encoding: {sys.stderr.encoding}")
    print(f"locale.getpreferredencoding(): {locale.getpreferredencoding()}")
    print(f"Default file system encoding: {sys.getfilesystemencoding()}")
    print("--- End: Force UTF-8 Globally ---")

force_utf8() # Execute the function to apply settings
# --- End: Force UTF-8 Globally ---

# === Start: Monkey Patch for SimplePropertyGraphStore Persistence ===
import json
import fsspec
from pathlib import Path
import logging

# 自定义 JSON 编码器，处理 set 类型和其他不可 JSON 序列化的类型
class CustomJSONEncoder(json.JSONEncoder):
    """处理 set 等特殊类型的 JSON 编码器"""
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)  # 将集合转换为列表
        # 可以在这里添加其他特殊类型的处理
        return super().default(obj)

def fixed_spgs_persist(self, persist_path: str, fs: fsspec.AbstractFileSystem = None) -> None:
    """为 SimplePropertyGraphStore 提供的增强 persist 方法，使用 UTF-8 编码。"""
    logger = logging.getLogger(__name__)
    logger.info(f"使用增强版 SimplePropertyGraphStore.persist 方法写入 {persist_path}")
    if fs is None:
        fs = fsspec.filesystem("file")

    dirpath = Path(persist_path).parent
    if not fs.exists(str(dirpath)):
            try:
                fs.makedirs(str(dirpath), exist_ok=True)
            except Exception as e:
                logger.error(f"使用 fsspec 创建目录 {dirpath} 失败: {e}")
                # 如果 fsspec 失败，尝试使用本地文件系统
                if fs.protocol == "file" and not dirpath.exists():
                    try:
                        dirpath.mkdir(parents=True, exist_ok=True)
                        logger.info(f"使用 pathlib 成功创建目录 {dirpath}。")
                    except Exception as e_inner:
                        logger.error(f"使用备用方法创建目录 {dirpath} 失败: {e_inner}")
                        raise

    try:
        # 获取图数据
        data_dict = self.to_dict()
    except AttributeError: # 如果 to_dict() 不是正确的方法
        logger.warning("SimplePropertyGraphStore 实例可能没有 'to_dict()' 方法。尝试 graph.model_dump()")
        try:
             # 基于原始错误跟踪的回退方法
             if hasattr(self, 'graph') and self.graph:
                 # 直接使用 UTF-8 编码写入文件
                 with fs.open(persist_path, "w", encoding='utf-8') as f:
                     f.write(self.graph.model_dump_json(indent=4))
                 logger.info(f"使用方法 (model_dump_json) 成功保存图存储到 {persist_path}")
                 return # 保存成功后退出
             else:
                 raise ValueError("在 SimplePropertyGraphStore 中未找到 graph 对象或为空。")
        except Exception as e_dump:
             logger.error(f"使用 model_dump_json 获取/导出图数据失败: {e_dump}")
             raise
    except Exception as e:
            logger.error(f"从 SimplePropertyGraphStore 获取图数据失败: {e}")
            raise

    # 如果 to_dict() 成功则执行此部分
    try:
        with fs.open(persist_path, "w", encoding='utf-8') as f:
            # 使用自定义编码器处理 set 类型，ensure_ascii=False 以直接写入 UTF-8
            json.dump(data_dict, f, ensure_ascii=False, indent=4, cls=CustomJSONEncoder)
        logger.info(f"使用方法 (to_dict) 成功保存图存储到 {persist_path}")
    except IOError as e:
        logger.error(f"写入 {persist_path} 失败: {e}")
        raise
    except Exception as e:
            logger.error(f"保存过程中遇到意外错误 {persist_path}: {e}")
            raise

# Apply the patch
# We need to ensure SimpleLabeledGraphStore is imported *before* this line
# Let's move the actual patching line down after the imports are done.

# === End: Monkey Patch ===

# --- Original Imports Start Here ---
import logging
import openai
# from dotenv import load_dotenv # Removed
from pathlib import Path
import json
from app.backend.config import config_service

# Ensure backend directory is in path for potential sibling imports if needed later
SCRIPT_DIR_PATH = Path(__file__).parent.resolve()
BACKEND_DIR_PATH = SCRIPT_DIR_PATH.parent

# --- LlamaIndex Imports ---
# Attempt imports and provide guidance if missing
try:
    from llama_index.core import (
        SimpleDirectoryReader,
        StorageContext,
        PropertyGraphIndex,
        Settings,
        load_index_from_storage, # Needed to check if index exists
    )
    from llama_index.core.graph_stores import SimpleGraphStore
    from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore  # 修改为正确的类名
    # from llama_index.core.graph_stores.types import GraphStoreData # Removed import
    from llama_index.llms.openai import OpenAI
    # from llama_index.embeddings.openai import OpenAIEmbedding # Keep commented unless strictly needed by chosen extractors/config
    from llama_index.core.node_parser import SentenceSplitter
    # from llama_index.extractors.entity import EntityExtractor # Keep commented unless using specific entity extraction

    # --- Apply the Monkey Patch AFTER successful import ---
    _original_persist = SimplePropertyGraphStore.persist  # 修改为正确的类名
    SimplePropertyGraphStore.persist = fixed_spgs_persist  # 修改为正确的类名
    print("成功为 SimplePropertyGraphStore.persist 应用补丁")
    
    # --- 添加针对 from_persist_path 方法的猴子补丁 ---
    # 保存原始方法
    _original_from_persist_path = SimplePropertyGraphStore.from_persist_path
    
    @classmethod
    def fixed_from_persist_path(cls, persist_path: str, fs=None):
        """确保使用 UTF-8 编码读取存储文件的修复版 from_persist_path 方法"""
        logger = logging.getLogger(__name__)
        logger.info(f"使用增强版 SimplePropertyGraphStore.from_persist_path 方法读取 {persist_path}")
        
        if fs is None:
            fs = fsspec.filesystem("file")
            
        try:
            # 明确指定使用 UTF-8 编码打开文件
            with fs.open(persist_path, "r", encoding="utf-8") as f:
                data = json.loads(f.read())
            
            # 调用原始方法中的后续处理逻辑
            store = cls()
            # 根据 LlamaIndex 源码中的实现，重构该方法的后续处理
            store.from_dict(data)
            logger.info(f"成功使用 UTF-8 编码读取索引文件 {persist_path}")
            return store
        except Exception as e:
            logger.error(f"从 {persist_path} 读取图存储数据失败: {e}")
            raise
    
    # 应用猴子补丁
    SimplePropertyGraphStore.from_persist_path = fixed_from_persist_path
    print("成功为 SimplePropertyGraphStore.from_persist_path 应用补丁")
    # --- End Apply Patch ---

except ImportError as e:
    print(f"ImportError: {e}. Please install required LlamaIndex libraries:")
    print("pip install llama-index llama-index-llms-openai") # Add others if needed
    sys.exit(1)
except Exception as e:
    # Catch potential errors during patching itself
    print(f"Error during LlamaIndex import or monkey-patching: {e}")
    sys.exit(1)

# --- Configuration ---
# ENV_PATH = BACKEND_DIR_PATH / ".env" # Removed

# print(f"Loading environment variables from: {ENV_PATH}") # Removed
# if not ENV_PATH.exists(): # Removed
#     print(f"Warning: .env file not found at {ENV_PATH}. Relying on existing environment variables.") # Removed
# load_dotenv(dotenv_path=ENV_PATH) # Removed

# Paths from config_service, resolved relative to BACKEND_DIR_PATH
DATA_SOURCE_DIR = (BACKEND_DIR_PATH / config_service.settings.DATA_SOURCE_DIR_RELATIVE).resolve()
LLAMA_GRAPH_INDEX_DIR = (BACKEND_DIR_PATH / config_service.settings.LLAMA_GRAPH_INDEX_DIR_RELATIVE).resolve()

# Ensure output directory exists
LLAMA_GRAPH_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# OpenAI Config
OPENAI_API_KEY_SECRET = config_service.settings.OPENAI_API_KEY
if not OPENAI_API_KEY_SECRET: # Keep this check as API key is crucial
    raise ValueError("OPENAI_API_KEY not found in config_service.settings.")
OPENAI_API_KEY = OPENAI_API_KEY_SECRET.get_secret_value() if OPENAI_API_KEY_SECRET else None


# LLM for extraction
LLAMA_EXTRACTION_LLM_MODEL = config_service.settings.LLAMA_EXTRACTION_LLM_MODEL
# Embedding model (LlamaIndex defaults often require one, even if retrieval avoids it)
OPENAI_EMBEDDING_MODEL = config_service.settings.OPENAI_EMBEDDING_MODEL

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- LlamaIndex Settings ---
# Configure LLM and Embedding model for LlamaIndex globally
# LlamaIndex often picks up OPENAI_API_KEY automatically if set in env
try:
    Settings.llm = OpenAI(model=LLAMA_EXTRACTION_LLM_MODEL, api_key=OPENAI_API_KEY)
    # Explicitly set embedding model if needed, otherwise LlamaIndex might default
    # Settings.embed_model = OpenAIEmbedding(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY) # Uncomment if needed
    logger.info(f"LlamaIndex Settings configured with LLM: {LLAMA_EXTRACTION_LLM_MODEL}")
except Exception as e:
    logger.error(f"Failed to configure LlamaIndex Settings: {e}")
    sys.exit(1)


# Chunking settings (can be customized)
Settings.chunk_size = 512
Settings.chunk_overlap = 50

# --- Main Logic ---
def create_graph_index(source_dir=None, index_dir=None):
    global DATA_SOURCE_DIR, LLAMA_GRAPH_INDEX_DIR
    logger.info(f"Starting LlamaIndex PropertyGraphIndex creation/update...")
    logger.info(f"Data Source Directory: {DATA_SOURCE_DIR}")
    logger.info(f"Index Directory: {LLAMA_GRAPH_INDEX_DIR}")
    logger.info(f"Using LLM for extraction: {LLAMA_EXTRACTION_LLM_MODEL}")

    if source_dir is not None:
        DATA_SOURCE_DIR = Path(source_dir)
    if index_dir is not None:
        LLAMA_GRAPH_INDEX_DIR = Path(index_dir)
        LLAMA_GRAPH_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_SOURCE_DIR.exists() or not any(DATA_SOURCE_DIR.iterdir()):
        logger.error(f"Data source directory not found or is empty: {DATA_SOURCE_DIR}")
        return

    try:
        # --- Check if index already exists ---
        try:
            # 使用直接的 SimplePropertyGraphStore 实例，而不是自定义类
            # 我们已经对 SimplePropertyGraphStore 应用了猴子补丁，确保它使用 UTF-8
            graph_store = SimplePropertyGraphStore()
            storage_context_load = StorageContext.from_defaults(
                 persist_dir=str(LLAMA_GRAPH_INDEX_DIR),
                 graph_store=graph_store # 传递正确类型的图存储实例
            )
            # 尝试使用 storage_context 加载已有索引
            existing_index = load_index_from_storage(storage_context_load)
            logger.info("已发现 PropertyGraphIndex。更新/刷新逻辑较复杂，暂未实现。")
            logger.info("如需从头重建，请删除索引目录并重新运行:")
            logger.info(f"  rm -rf \"{LLAMA_GRAPH_INDEX_DIR}\"")
            logger.info("跳过索引创建。")
            # 注意: 实现稳健的索引更新（刷新）并不简单
            # 它涉及跟踪更改的文档、删除旧节点/关系、添加新节点等。
            # 目前，我们只关注在索引不存在时创建它。
            return existing_index # 如果索引存在则退出
        except FileNotFoundError:
            logger.info("未找到现有索引（或 graph_store.json 缺失）。继续创建。")
        except Exception as load_err:
            logger.warning(f"检查现有索引时出错（可能继续创建）: {load_err}")
            # 决定是否应该继续或在通用加载错误时中止

        # --- Load Documents ---
        logger.info("从源目录加载文档...")
        reader = SimpleDirectoryReader(str(DATA_SOURCE_DIR), recursive=True) # 使用字符串路径
        documents = reader.load_data()
        if not documents:
            logger.error("未找到或加载文档。中止。")
            return
        logger.info(f"已加载 {len(documents)} 份文档。")

        # --- Setup Graph Store and Storage Context for Creation ---
        # 使用直接的 SimplePropertyGraphStore 实例，而不是自定义类 PersistableSimpleGraphStore
        graph_store_create = SimplePropertyGraphStore()
        storage_context_create = StorageContext.from_defaults(graph_store=graph_store_create)

        # --- Build the PropertyGraphIndex ---
        logger.info("构建 PropertyGraphIndex（这可能需要时间并调用 LLM）...")
        # 这一步使用全局配置的 Settings.llm 等。
        index = PropertyGraphIndex.from_documents(
            documents,
            storage_context=storage_context_create,
            show_progress=True,
            # include_embeddings=False # 如果需要，可以试验，但可能会打破默认设置
        )

        # --- Persist the index ---
        logger.info(f"将索引持久化到 {LLAMA_GRAPH_INDEX_DIR}...")
        # 创建期间使用的存储上下文包含要持久化的数据
        index.storage_context.persist(persist_dir=str(LLAMA_GRAPH_INDEX_DIR)) # 使用字符串路径

        logger.info("LlamaIndex PropertyGraphIndex 创建完成。")
        return index

    except ImportError as e:
        logger.error(f"索引创建期间的 ImportError: {e}。确保已安装 LlamaIndex。")
    except Exception as e:
        logger.exception(f"LlamaIndex 图索引创建期间出错: {e}")

if __name__ == "__main__":
    create_graph_index() 