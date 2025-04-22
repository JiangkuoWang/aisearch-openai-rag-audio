# app/backend/rag_providers/llama_index_graph.py
import locale
import os
import sys
import codecs
import json
import fsspec
from pathlib import Path

# --- Start: Force UTF-8 Globally (Place at the very top) ---
ENCODING = "utf-8"

def force_utf8():
    # This function might be slightly redundant if both scripts run in the same process
    # But it ensures UTF-8 is set if this module is imported/run independently.
    print(f"[{__name__}] Attempting to force Python default encoding to {ENCODING}...")
    try:
        os.environ['LANG'] = f'en_US.{ENCODING}'
        os.environ['LC_ALL'] = f'en_US.{ENCODING}'
    except Exception:
        pass # Ignore errors setting env vars here, might be set already
    try:
        locale.setlocale(locale.LC_ALL, f'en_US.{ENCODING}')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, '') # Use system default
            if codecs.lookup(locale.getpreferredencoding()).name != ENCODING:
                utf8_locale = ""
                if sys.platform == 'win32':
                    locales_to_try = ['.UTF-8', '.65001']
                else:
                    locales_to_try = ['C.UTF-8', 'en_US.UTF-8']
                for loc in locales_to_try:
                    try:
                        locale.setlocale(locale.LC_ALL, loc)
                        utf8_locale = loc
                        break
                    except locale.Error:
                        continue
                if not utf8_locale:
                    print(f"[{__name__}] Warning: Could not set a supported UTF-8 locale.")
        except Exception:
            print(f"[{__name__}] Warning: Failed to set locale.")
    try:
        os.environ["PYTHONIOENCODING"] = ENCODING
    except Exception:
        pass # Ignore errors setting env vars here
    try:
        if sys.stdout.encoding != ENCODING:
             sys.stdout = codecs.getwriter(ENCODING)(sys.stdout.buffer, 'replace')
        if sys.stderr.encoding != ENCODING:
             sys.stderr = codecs.getwriter(ENCODING)(sys.stderr.buffer, 'replace')
    except Exception as e:
        print(f"[{__name__}] Warning: Failed to reconfigure stdout/stderr: {e}")
    # Verify effective encoding - REMOVED problematic lines
    # The following lines would cause AttributeError after reconfiguration
    # print(f"[{__name__}] sys.stdout.encoding: {sys.stdout.encoding}") # Line 51 commented out
    # print(f"[{__name__}] sys.stderr.encoding: {sys.stderr.encoding}") # Line 52 commented out
    print(f"[{__name__}] locale.getpreferredencoding(): {locale.getpreferredencoding()}")
    print(f"[{__name__}] --- End: Force UTF-8 Globally ---")

force_utf8()
# --- End: Force UTF-8 Globally ---

# --- Original Imports Start Here ---
import logging
from pathlib import Path
import openai
import asyncio
from typing import List, Dict, Any, Optional
# import sys # Already imported
import json # Import json
# import os # Already imported
# import codecs # Already imported
# import locale # Already imported

# Ensure backend directory is in path for sibling imports
BACKEND_DIR_PATH = Path(__file__).parent.parent.resolve()
if str(BACKEND_DIR_PATH) not in sys.path:
    sys.path.append(str(BACKEND_DIR_PATH))

from rag_providers.base import BaseRAGProvider

# --- LlamaIndex Imports ---
try:
    from llama_index.core import (
        StorageContext,
        PropertyGraphIndex,
        load_index_from_storage,
        Settings,
        QueryBundle,
    )
    # Import necessary components for the correct graph store
    from llama_index.core.graph_stores import SimpleGraphStore
    from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore  # 使用正确的类名
    from llama_index.llms.openai import OpenAI
    # from llama_index.core.indices.property_graph import PropertyGraphStore # Already imported SimpleGraphStore
except ImportError as e:
    print(f"ImportError: {e}. Please install required LlamaIndex libraries:")
    print("pip install llama-index llama-index-llms-openai") # Add others if needed
    sys.exit(1)

logger = logging.getLogger(__name__)

# --- 猴子补丁：为 SimplePropertyGraphStore 的 persist 方法强制使用 UTF-8 ---
import fsspec

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
    if not dirpath.exists():
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

# 应用猴子补丁
_original_persist = SimplePropertyGraphStore.persist
SimplePropertyGraphStore.persist = fixed_spgs_persist
print("成功为 SimplePropertyGraphStore.persist 应用补丁")

# --- 添加猴子补丁：为 SimplePropertyGraphStore 的 from_persist_path 方法强制使用 UTF-8 ---
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
        # 通常情况下会使用数据创建对象实例
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

# --- LlamaIndexGraphRAGProvider Class ---
class LlamaIndexGraphRAGProvider(BaseRAGProvider):
    """RAG provider using a pre-built LlamaIndex PropertyGraphIndex."""

    def __init__(self,
                 openai_client: openai.OpenAI, # Keep for potential future use or config
                 index_dir: Path,
                 embedding_model_name: str, # Keep for potential future use or config
                 llm_model_name: str # Pass LLM model name for retriever
                 ):
        self.index_dir = index_dir
        self.index: Optional[PropertyGraphIndex] = None
        self.retriever = None # Will be set during initialization
        self.openai_client = openai_client
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        logger.info(f"LlamaIndexGraphRAGProvider initialized with index directory: {self.index_dir}")

    async def initialize(self) -> None:
        """Load the LlamaIndex PropertyGraphIndex from disk."""
        if not self.index_dir.is_dir():
            logger.error(f"LlamaIndex graph index directory not found: {self.index_dir}")
            raise FileNotFoundError(f"LlamaIndex graph index directory not found: {self.index_dir}")

        logger.info(f"Loading LlamaIndex PropertyGraphIndex from {self.index_dir}...")
        try:
            # 使用兼容的模型名称，而不是 self.llm_model_name
            # self.llm_model_name 可能是 gpt-4o-realtime-preview，这不被 LlamaIndex 支持
            # 使用 gpt-4o 或 gpt-4 作为替代
            llm_compatible_model_name = "gpt-4o"
            llm = OpenAI(model=llm_compatible_model_name)

            # 使用 SimplePropertyGraphStore 类（已应用猴子补丁确保 UTF-8 编码）
            graph_store = SimplePropertyGraphStore()
            # 检查特定的存储文件是否存在，可能比仅检查目录更可靠
            graph_store_path = self.index_dir / "graph_store.json"
            if not graph_store_path.exists():
                logger.warning(f"graph_store.json not found in {self.index_dir}. StorageContext might load defaults.")
                # 如果图存储是必需的且未找到，可能会引发错误
                # 现在，继续并让 load_index_from_storage 处理它

            storage_context = StorageContext.from_defaults(
                persist_dir=str(self.index_dir),
                graph_store=graph_store # Pass the custom store instance
            )

            # Load the main index object
            self.index = load_index_from_storage(storage_context, llm=llm)

            if not isinstance(self.index, PropertyGraphIndex):
                 raise TypeError(f"Loaded index is not a PropertyGraphIndex, found {type(self.index)}")

            # --- Configure Retriever ---
            self.retriever = self.index.as_retriever()
            logger.info("LlamaIndex PropertyGraphIndex loaded successfully.")
            logger.info(f"Retriever configured: {type(self.retriever).__name__}")

        except Exception as e:
            logger.exception(f"Failed to load LlamaIndex graph index: {e}")
            self.index = None
            self.retriever = None
            raise # Re-raise the exception

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using the configured LlamaIndex retriever."""
        if not self.retriever:
            logger.error("LlamaIndex retriever not initialized. Cannot search.")
            return []
        if not query:
             logger.error("Search called without a query.")
             return []

        logger.info(f"Searching LlamaIndex (Graph?) for '{query}' (top_k={top_k})...")

        try:
            # Use the retriever (potentially graph-aware)
            retrieved_nodes = await self.retriever.aretrieve(query)

            results = []
            if not retrieved_nodes:
                logger.info("LlamaIndex search returned no results.")
                return []

            logger.info(f"LlamaIndex search returned {len(retrieved_nodes)} nodes.")

            # --- Format LlamaIndex Nodes into expected Dict format ---
            # Adapt formatting based on what the retriever returns (Nodes, text chunks, graph paths)
            for i, node_with_score in enumerate(retrieved_nodes):
                node = node_with_score.node
                # Use node ID as chunk_id for grounding reference
                chunk_id = node.node_id
                # Get node content - might be raw text or structured graph info
                # Using metadata_mode='all' might include relationships
                text = node.get_content(metadata_mode="llm") # Prioritize base text

                # Include relationships in text if retriever mode brings them?
                # Example: If text is just base node, maybe add relations from metadata?
                # Check node.metadata structure after retrieval
                # if 'relationships' in node.metadata:
                #     text += "\nRelationships: " + str(node.metadata['relationships'])

                # Try to get source document filename
                title = node.metadata.get("file_name", "Graph Context")

                # Use LlamaIndex score if available, otherwise a placeholder
                score = node_with_score.score if node_with_score.score is not None else 1.0

                results.append({
                    "chunk_id": chunk_id,
                    "text": text.strip(), # Clean up whitespace
                    "title": title,
                    "score": float(score)
                })
                # Respect top_k limit
                if len(results) >= top_k:
                     break

            logger.info(f"Formatted {len(results)} results for LLM.")
            return results

        except Exception as e:
            logger.exception(f"Error during LlamaIndex search: {e}")
            return []

    async def get_details(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve details for specific LlamaIndex node IDs."""
        if not self.index or not hasattr(self.index, 'docstore'):
             logger.error("LlamaIndex index or docstore not available for get_details.")
             return []

        logger.info(f"Getting details for {len(chunk_ids)} LlamaIndex node IDs: {chunk_ids}")
        docs = []
        found_ids = set()

        # Node IDs from LlamaIndex are typically strings
        for node_id in chunk_ids:
            if not isinstance(node_id, str) or node_id in found_ids:
                logger.warning(f"Skipping invalid or duplicate node ID: {node_id}")
                continue

            try:
                 # Retrieve node directly from the docstore using the node_id
                 node = self.index.docstore.get_node(node_id, raise_error=False)

                 if node:
                     content = node.get_content(metadata_mode="llm") # Get the core text content
                     title = node.metadata.get("file_name", "Node Detail") # Get original filename
                     docs.append({
                         "chunk_id": node_id,
                         "title": title,
                         "chunk": content.strip() # Match expected key 'chunk'
                     })
                     found_ids.add(node_id)
                 else:
                     logger.warning(f"Node ID '{node_id}' not found in LlamaIndex docstore for grounding.")

            except Exception as e:
                 logger.warning(f"Error retrieving details for LlamaIndex node ID '{node_id}': {e}")
                 # Continue to next ID

        logger.info(f"Retrieved details for {len(docs)} LlamaIndex node IDs.")
        return docs 