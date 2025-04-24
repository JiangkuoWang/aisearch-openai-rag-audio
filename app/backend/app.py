import logging
import os
from pathlib import Path
import json
import numpy as np
import openai
import sys
from typing import Any, List, Callable, Optional, Union, Dict
import asyncio
import tempfile
import shutil # Added for directory cleanup new_pro

from aiohttp import web
from dotenv import load_dotenv

# Local imports
from rtmt import RTMiddleTier
from rag_providers.base import BaseRAGProvider
from rag_providers.in_memory import InMemoryRAGProvider
from rag_providers.llama_index_graph import LlamaIndexGraphRAGProvider
from ragtools import attach_rag_tools
# Ensure rag_upload_utils.py exists in the same directory
try:
    from rag_upload_utils import extract_text, chunk_text
except ImportError:
    logging.error("Could not import from rag_upload_utils.py. Make sure the file exists and is correct.")
    # Define dummy functions to prevent crashing if import fails
    def extract_text(filename: str, raw: bytes) -> str: return ""
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]: return []


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voicerag")

# Define backend directory for constructing absolute paths
BACKEND_DIR = Path(__file__).parent.resolve()

# --- Helper Function to Update RAG Provider ---
def update_rag_provider(app: web.Application, rag_provider: Optional[BaseRAGProvider]):
    """Dynamically attach/detach RAG tools based on the provider."""
    rtmt = app.get("rtmt")
    if not rtmt:
        logger.error("RTMiddleTier instance not found in app context.")
        return

    # Remove existing RAG tools first
    for name in ("search", "report_grounding"):
        if name in rtmt.tools:
            del rtmt.tools[name]
            logger.debug(f"Removed RAG tool: {name}")

    # Attach new tools if a provider is given
    if rag_provider:
        try:
            attach_rag_tools(rtmt, rag_provider)
            logger.info(f"Attached RAG tools using provider: {type(rag_provider).__name__}")
        except Exception as e:
            logger.exception(f"Error attaching RAG tools: {e}")
    else:
        logger.info("Detached RAG tools (provider is None).")

    app["rag_provider"] = rag_provider # Store the current provider

# --- HTTP Handlers ---
async def handle_rag_config(request: web.Request):
    """Handles POST request to set the RAG provider type."""
    try:
        data = await request.json()
        provider_type = data.get("provider_type", "").lower()
        if provider_type not in ("none", "in_memory", "llama_index"):
            logger.warning(f"Invalid provider_type received: {provider_type}")
            return web.HTTPBadRequest(text="Invalid provider_type specified. Must be 'none', 'in_memory', or 'llama_index'.")

        app = request.app
        app["rag_provider_type"] = provider_type
        logger.info(f"RAG provider type set to: {provider_type}")

        # If 'none', detach tools immediately.
        if provider_type == "none":
            update_rag_provider(app, None)
        # For other types, we wait for an upload to activate the provider.
        # If a provider of the *same type* was already active, it remains until replaced by upload.
        # If the type changes, we could optionally clear the old provider here:
        # else:
        #     if app.get("rag_provider") and type(app.get("rag_provider")).__name__.lower().replace("ragprovider","") != provider_type.replace("_",""):
        #          update_rag_provider(app, None) # Clear if type differs

        return web.json_response({"status": "ok", "message": f"RAG provider type set to {provider_type}"})
    except json.JSONDecodeError:
        logger.error("Failed to decode JSON in /rag-config request.")
        return web.HTTPBadRequest(text="Invalid JSON payload.")
    except Exception as e:
        logger.exception(f"Error handling RAG config: {e}")
        return web.HTTPInternalServerError(text="Error processing RAG configuration.")

async def handle_upload(request: web.Request):
    """Handles file uploads to build and activate a RAG provider."""
    app = request.app
    provider_type = app.get("rag_provider_type")

    if provider_type not in ("in_memory", "llama_index"):
        logger.warning(f"Upload attempt received but RAG type is '{provider_type}'. Ignoring.")
        return web.HTTPBadRequest(text=f"File upload not supported for RAG type '{provider_type}'. Select 'in_memory' or 'llama_index' first via /rag-config.")

    # Create a temporary directory for uploaded files
    temp_dir = Path(tempfile.mkdtemp(prefix="rag_upload_"))
    logger.info(f"Created temporary directory for uploads: {temp_dir}")
    try:
        reader = await request.multipart()
        texts: List[str] = []
        titles: List[str] = []
        file_paths: List[Path] = [] # Store paths for LlamaIndex

        part_index = 0
        while True:
            part = await reader.next()
            if part is None:
                break # End of parts
            if part.filename:
                # Sanitize filename if needed, here we just use it
                safe_filename = f"{part_index}_{Path(part.filename).name}"
                temp_file_path = temp_dir / safe_filename
                logger.info(f"Processing uploaded file: {part.filename} -> {temp_file_path}")
                file_size = 0
                try:
                    with open(temp_file_path, "wb") as f:
                        while True:
                            chunk = await part.read_chunk(size=8192) # Read in chunks
                            if not chunk:
                                break
                            f.write(chunk)
                            file_size += len(chunk)
                    logger.info(f"Saved {part.filename} ({file_size} bytes) to {temp_file_path}")
                    file_paths.append(temp_file_path) # Store path for later use

                    # Extract and chunk text immediately after saving
                    raw_data = temp_file_path.read_bytes()
                    try:
                        extracted_text = extract_text(part.filename, raw_data)
                        if not extracted_text:
                             logger.warning(f"No text extracted from {part.filename}, skipping.")
                             continue # Skip this file
                             
                        chunks = chunk_text(extracted_text) # Use default chunk size
                        if not chunks:
                            logger.warning(f"No chunks created from {part.filename}, skipping.")
                            continue # Skip this file
                            
                        logger.info(f"Extracted and chunked {part.filename} into {len(chunks)} chunks.")

                        # Append chunks and their titles
                        texts.extend(chunks)
                        titles.extend([part.filename] * len(chunks))
                    except ImportError as e:
                        # 明确捕获缺少依赖库的错误
                        logger.error(f"Missing required library for processing {part.filename}: {e}")
                        return web.HTTPInternalServerError(text=f"Missing required library for processing {part.filename}. Please install all required dependencies.")
                    except Exception as e:
                        logger.exception(f"Error extracting text from {part.filename}: {e}")
                        # 继续处理其他文件，不中断上传过程

                except Exception as e:
                    logger.exception(f"Error processing/saving file {part.filename}: {e}")
                    # Decide if one failed file should abort the whole upload
                    # return web.HTTPInternalServerError(text=f"Error processing file {part.filename}")
                part_index += 1
            else:
                # Handle non-file parts if necessary, e.g., form fields
                 logger.warning("Received a multipart part without a filename, ignoring.")


        if not texts:
            logger.error("No text could be extracted or chunked from uploaded files.")
            # Clean up temp dir as nothing was processed
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up empty temporary directory: {temp_dir}")
            return web.HTTPBadRequest(text="No processable content found in uploaded files.")

        logger.info(f"Total text chunks to process: {len(texts)}")

        # Get necessary components from app context
        openai_client = app.get("openai_client")
        embedding_model = app.get("openai_embedding_model")
        if not openai_client or not embedding_model:
             logger.error("OpenAI client or embedding model not found in app context.")
             return web.HTTPInternalServerError(text="Backend configuration error.")

        logger.info(f"Generating embeddings using model: {embedding_model}...")
        try:
            # Ensure client is async if using await
            if not isinstance(openai_client, openai.AsyncOpenAI):
                 logger.error("OpenAI client is not async, cannot await.")
                 # Fallback or error - assuming sync client for now if needed, but should be async
                 # For demo, let's assume it *is* async as intended
                 return web.HTTPInternalServerError(text="Internal configuration error (OpenAI client type).")

            # Batch embedding generation
            response = await openai_client.embeddings.create(input=texts, model=embedding_model)
            vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
            logger.info(f"Generated {vectors.shape[0]} embeddings with dimension {vectors.shape[1]}.")
        except Exception as e:
             logger.exception("Error generating embeddings.")
             return web.HTTPInternalServerError(text="Failed to generate embeddings for uploaded content.")

        # Prepare metadata list matching the order of texts/vectors
        metadata_list = [
            {"chunk_id": f"{titles[i]}-{i}", "text": texts[i], "title": titles[i]}
            for i in range(len(texts))
        ]

        new_provider: Optional[BaseRAGProvider] = None
        if provider_type == "in_memory":
            logger.info("Initializing InMemoryRAGProvider...")
            try:
                new_provider = InMemoryRAGProvider(
                    openai_client=openai_client, # Pass the client object
                    embedding_model=embedding_model,
                    all_metadata=metadata_list,
                    all_vectors=vectors
                )
                logger.info("InMemoryRAGProvider initialized.")
            except Exception as e:
                 logger.exception("Failed to initialize InMemoryRAGProvider")
                 return web.HTTPInternalServerError(text="Failed to initialize In-Memory RAG provider.")

        elif provider_type == "llama_index":
            from scripts.create_llama_graph_index import create_graph_index
            llama_index_persist_dir = temp_dir / "llama_index_data"
            llama_index_persist_dir.mkdir(exist_ok=True)    
            index = await asyncio.to_thread(
                create_graph_index,
                source_dir=str(temp_dir),
                index_dir=str(llama_index_persist_dir)
            )
            # logger.info("Initializing LlamaIndexGraphRAGProvider...")
            # llama_index_persist_dir = temp_dir / "llama_index_data"
            # llama_index_persist_dir.mkdir(exist_ok=True)
            # logger.info(f"LlamaIndex persistence directory: {llama_index_persist_dir}")

            # try:
            #     # 1. 正确导入所需组件 index
            #     from llama_index.core import SimpleDirectoryReader, Document
            #     from llama_index.core.node_parser import SentenceSplitter
            #     from llama_index.core.storage import StorageContext
            #     from llama_index.core.prompts import PromptTemplate
            #     from llama_index.llms.openai import OpenAI as LlamaOpenAI
            #     from llama_index.core.schema import TransformComponent, BaseNode
            #     import json
            #     import re

            #     # 2. 设置LLM
            #     # llm_model_name = app["openai_model"]
            #     llm_model_name = "gpt-4o"
            #     llm = LlamaOpenAI(model=llm_model_name)
            #     logger.info(f"初始化LlamaOpenAI，使用模型: {llm_model_name}")

            #     # 3. 加载文档
            #     logger.info(f"从{temp_dir}加载文档")
            #     docs = SimpleDirectoryReader(str(temp_dir)).load_data()
            #     logger.info(f"成功加载{len(docs)}个文档")

            #     # 4. 分割文档为节点
            #     splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
            #     nodes = splitter.get_nodes_from_documents(docs)
            #     logger.info(f"将文档分割为{len(nodes)}个节点")

            #     # 5. 创建三元组提取模板和解析函数
            #     KG_TRIPLET_EXTRACT_TMPL = """
            #     -目标-
            #     给定文本文档，识别文本中的所有实体及其实体类型，以及已识别实体之间的所有关系。
            #     从文本中提取多达{max_knowledge_triplets}个实体关系三元组。

            #     -步骤-
            #     1. 识别所有实体。对于每个识别的实体，提取以下信息：
            #     - entity_name: 实体名称，首字母大写
            #     - entity_type: 实体类型
            #     - entity_description: 实体属性和活动的全面描述

            #     2. 从步骤1中识别的实体中，识别所有彼此明确相关的(source_entity, target_entity)对。
            #     对于每对相关实体，提取以下信息：
            #     - source_entity: 源实体的名称，如步骤1中所识别
            #     - target_entity: 目标实体的名称，如步骤1中所识别
            #     - relation: 源实体和目标实体之间的关系
            #     - relationship_description: 解释为什么您认为源实体和目标实体相互关联

            #     3. 输出格式：
            #     - 以有效的JSON格式返回结果，包含两个键：'entities'（实体对象列表）和'relationships'（关系对象列表）。
            #     - 排除JSON结构之外的任何文本（例如，无解释或评论）。
            #     - 如果未识别到实体或关系，则返回空列表：{{ "entities": [], "relationships": [] }}。

            #     -实际数据-
            #     ######################
            #     text: {text}
            #     ######################
            #     output:"""

            #     def parse_fn(response_str):
            #         """解析LLM响应，提取实体和关系"""
            #         json_pattern = r"\{{.*\}}"
            #         match = re.search(json_pattern, response_str, re.DOTALL)
            #         entities = []
            #         relationships = []
            #         if not match:
            #             return entities, relationships
            #         json_str = match.group(0)
            #         try:
            #             data = json.loads(json_str)
            #             entities = [
            #                 (
            #                     entity["entity_name"],
            #                     entity["entity_type"],
            #                     entity["entity_description"],
            #                 )
            #                 for entity in data.get("entities", [])
            #             ]
            #             relationships = [
            #                 (
            #                     relation["source_entity"],
            #                     relation["target_entity"],
            #                     relation["relation"],
            #                     relation["relationship_description"],
            #                 )
            #                 for relation in data.get("relationships", [])
            #             ]
            #             return entities, relationships
            #         except json.JSONDecodeError as e:
            #             logger.error(f"解析JSON错误: {e}")
            #             return entities, relationships

            #     # 6. 创建自定义图关系提取器
            #     class GraphRAGExtractor(TransformComponent):
            #         """提取图形三元组的组件"""
            #         llm: Any  # 声明字段
            #         extract_prompt: PromptTemplate
            #         parse_fn: Callable
            #         max_paths_per_chunk: int = 3  # 带默认值的字段
                    
                
            #         def __call__(self, nodes, **kwargs):
            #             """处理节点列表"""
            #             for node in nodes:
            #                 # 从节点获取文本内容
            #                 text = node.get_content()
            #                 try:
            #                     # 使用LLM提取三元组
            #                     llm_response = self.llm.predict(
            #                         self.extract_prompt,
            #                         text=text,
            #                         max_knowledge_triplets=self.max_paths_per_chunk
            #                     )
            #                     # 解析LLM响应
            #                     entities, relationships = self.parse_fn(llm_response)
                                
            #                     # 添加到节点元数据
            #                     from llama_index.core.graph_stores.types import EntityNode, Relation, KG_NODES_KEY, KG_RELATIONS_KEY
                                
            #                     # 处理实体
            #                     entity_nodes = []
            #                     for entity, entity_type, description in entities:
            #                         entity_metadata = node.metadata.copy()
            #                         entity_metadata["entity_description"] = description
            #                         entity_node = EntityNode(
            #                             name=entity,
            #                             label=entity_type,
            #                             properties=entity_metadata
            #                         )
            #                         entity_nodes.append(entity_node)
                                
            #                     # 处理关系
            #                     relation_nodes = []
            #                     for subj, obj, rel, description in relationships:
            #                         relation_metadata = node.metadata.copy()
            #                         relation_metadata["relationship_description"] = description
            #                         relation_node = Relation(
            #                             label=rel,
            #                             source_id=subj,
            #                             target_id=obj,
            #                             properties=relation_metadata
            #                         )
            #                         relation_nodes.append(relation_node)
                                
            #                     # 保存到节点元数据
            #                     node.metadata[KG_NODES_KEY] = entity_nodes
            #                     node.metadata[KG_RELATIONS_KEY] = relation_nodes
                                
            #                 except Exception as e:
            #                     logger.error(f"处理节点时出错: {e}")
                        
            #             return nodes

            #     # 7. 创建图关系提取器实例
            #     kg_extractor = GraphRAGExtractor(
            #         llm=llm,
            #         extract_prompt=PromptTemplate(KG_TRIPLET_EXTRACT_TMPL),
            #         parse_fn=parse_fn,
            #         max_paths_per_chunk=3
            #     )
            #     logger.info("创建了GraphRAGExtractor实例")

            #     # 8. 从 create_llama_graph_index.py 导入必要的组件
            #     import sys
            #     from pathlib import Path
                
            #     # 添加 scripts 目录到系统路径
            #     scripts_dir = BACKEND_DIR / "scripts"
            #     if str(scripts_dir) not in sys.path:
            #         sys.path.append(str(scripts_dir))
            #         logger.info(f"已将 {scripts_dir} 添加到系统路径")
                
            #     try:
            #         # 导入 create_llama_graph_index 脚本中的必要组件
            #         from create_llama_graph_index import (
            #             force_utf8, 
            #             CustomJSONEncoder, 
            #             fixed_spgs_persist, 
            #             fixed_from_persist_path,
            #             SimplePropertyGraphStore,
            #             StorageContext,
            #             PropertyGraphIndex,
            #             load_index_from_storage,
            #             Settings,
            #             OpenAI,
            #             SimpleDirectoryReader
            #         )
                    
            #         # 应用 UTF-8 编码处理
            #         force_utf8()
                    
            #         # 应用猴子补丁
            #         original_persist = SimplePropertyGraphStore.persist
            #         SimplePropertyGraphStore.persist = fixed_spgs_persist
            #         original_from_persist_path = SimplePropertyGraphStore.from_persist_path
            #         SimplePropertyGraphStore.from_persist_path = fixed_from_persist_path
            #         logger.info("已应用 SimplePropertyGraphStore 编码补丁")
                    
            #         # 设置 LlamaIndex 全局配置
            #         Settings.llm = OpenAI(model="gpt-4o", api_key=openai_api_key)
            #         Settings.chunk_size = 512
            #         Settings.chunk_overlap = 50
            #         logger.info("已配置 LlamaIndex 全局设置")
                    
            #         # 9. 定义索引目录
            #         llama_index_persist_dir = Path(llama_index_persist_dir)  # 确保是 Path 对象
                    
            #         # 10. 定义异步建索引函数
            #         async def build_index_async():
            #             # 将文件从临时目录加载到文档对象
            #             logger.info(f"从 {temp_dir} 加载文档...")
                        
            #             # 定义同步函数在线程池中执行
            #             def create_index_sync():
            #                 logger.info("在线程池中开始构建 PropertyGraphIndex...")
                            
            #                 # 检查索引是否已存在
            #                 try:
            #                     if (llama_index_persist_dir / "graph_store.json").exists():
            #                         try:
            #                             # 使用直接的 SimplePropertyGraphStore 实例
            #                             graph_store = SimplePropertyGraphStore()
            #                             storage_context_load = StorageContext.from_defaults(
            #                                 persist_dir=str(llama_index_persist_dir),
            #                                 graph_store=graph_store
            #                             )
            #                             # 尝试使用 storage_context 加载已有索引
            #                             existing_index = load_index_from_storage(storage_context_load)
            #                             logger.info("已发现现有 PropertyGraphIndex，将使用它")
            #                             return existing_index
            #                         except Exception as load_err:
            #                             logger.warning(f"检查现有索引时出错: {load_err}")
            #                 except Exception as e:
            #                     logger.warning(f"检查索引文件时出错: {e}")
                            
            #                 # 加载文档
            #                 reader = SimpleDirectoryReader(str(temp_dir))
            #                 documents = reader.load_data()
            #                 if not documents:
            #                     logger.error("未找到或加载文档。中止。")
            #                     return None
            #                 logger.info(f"已加载 {len(documents)} 份文档。")
                            
            #                 # 设置图存储和存储上下文
            #                 graph_store_create = SimplePropertyGraphStore()
            #                 storage_context_create = StorageContext.from_defaults(graph_store=graph_store_create)
                            
            #                 # 构建 PropertyGraphIndex
            #                 logger.info("构建 PropertyGraphIndex（这可能需要时间并调用 LLM）...")
            #                 # 这一步使用全局配置的 Settings.llm 等。
            #                 index = PropertyGraphIndex.from_documents(
            #                     documents,
            #                     storage_context=storage_context_create,
            #                     show_progress=True,
            #                 )
                            
            #                 # 持久化索引
            #                 logger.info(f"将索引持久化到 {llama_index_persist_dir}...")
            #                 # 创建期间使用的存储上下文包含要持久化的数据
            #                 index.storage_context.persist(persist_dir=str(llama_index_persist_dir))
                            
            #                 logger.info("LlamaIndex PropertyGraphIndex 创建完成。")
            #                 return index
                        
            #             # 在线程池中执行索引构建
            #             return await asyncio.to_thread(create_index_sync)
                    
            #         # 开始异步构建索引
            #         index = await build_index_async()
                    
                    # 创建检索器
            retriever = index.as_retriever(similarity_top_k=5)
                    
                    # 实例化并设置provider
            new_provider = LlamaIndexGraphRAGProvider(
                        openai_client=openai_client,
                        index_dir=llama_index_persist_dir,
                        embedding_model_name=embedding_model,
                        llm_model_name=app["openai_model"],
                    )
                    
                    # 设置索引和检索器
            new_provider.index = index
            new_provider.retriever = retriever
            logger.info("LlamaIndexGraphRAGProvider已初始化，配置了PropertyGraphIndex和检索器")
            
            #     except ImportError as e:
            #         logger.error(f"导入错误: {e}，请安装必要的库: pip install llama-index llama-index-llms-openai")
            #         return web.HTTPInternalServerError(text=f"缺少必要的库: {e}")
            #     except Exception as e:
            #         logger.exception(f"初始化 LlamaIndex 组件时出错: {e}")
            #         return web.HTTPInternalServerError(text=f"初始化 LlamaIndex 组件时出错: {e}")

        # Activate the new provider
        if new_provider:
            update_rag_provider(app, new_provider)
            logger.info(f"Successfully activated {provider_type} RAG provider with uploaded data.")
            # Clean up temp dir *only* after successful provider activation
            # For LlamaIndex, decide if you need to keep the persisted index dir
            # but remove the original uploaded files.
            # Simple cleanup for now: remove the whole temp dir. Adjust if needed.
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
            return web.json_response({
                "status": "ok",
                "message": f"{provider_type} RAG provider activated with {len(texts)} chunks from {part_index} files."
            })
        else:
            logger.error("Failed to create a RAG provider instance after processing uploads.")
            # Clean up temp dir as provider creation failed
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory due to provider creation failure: {temp_dir}")
            return web.HTTPInternalServerError(text="Failed to create RAG provider instance.")

    except Exception as e:
        logger.exception(f"Unhandled error during file upload processing: {e}")
        # Ensure cleanup happens on any unexpected error
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.warning(f"Cleaned up temporary directory due to unhandled error: {temp_dir}")
        return web.HTTPInternalServerError(text="An unexpected error occurred during file upload processing.")


# --- Main Application Setup ---
async def create_app():
    app = web.Application()
    # Load .env file if not in production
    if not os.environ.get("RUNNING_IN_PRODUCTION"):
        logger.info("Running in development mode, loading from .env file")
        env_path = BACKEND_DIR / ".env"
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path)
            logger.info(f"Loaded environment variables from: {env_path}")
        else:
            logger.warning(f".env file not found at {env_path}, relying on existing environment variables.")

    # --- Load OpenAI Configuration ---
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_model = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview") # Ensure this is a valid model
    openai_embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large") # Ensure this is valid

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    # --- Initialize OpenAI Async Client and store in app context ---
    # Use Async Client for await operations like embeddings
    try:
        openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        app["openai_client"] = openai_client
        app["openai_model"] = openai_model
        app["openai_embedding_model"] = openai_embedding_model
        logger.info("AsyncOpenAI client initialized.")
    except Exception as e:
        logger.exception("Failed to initialize AsyncOpenAI client.")
        raise ValueError(f"Could not initialize OpenAI client: {e}")


    # --- RAG Configuration State --- Initialized Dynamically
    app["rag_provider_type"] = "none" # Default to no RAG
    app["rag_provider"] = None        # No provider active initially

    # --- Initialize RTMiddleTier and store in app context ---
    rtmt = RTMiddleTier(
        openai_api_key=openai_api_key,
        model=openai_model,
        voice_choice=os.environ.get("OPENAI_REALTIME_VOICE_CHOICE") or "alloy"
    )
    app["rtmt"] = rtmt

    # Configure System Prompt for RTMiddleTier
    # Ensure this prompt reflects the dynamic RAG capability if needed
    rtmt.system_message = """
        You are a helpful assistant. If the user has enabled the knowledge base search, only answer questions based on information found using the 'search' tool. Otherwise, answer normally.
        The user is listening to answers with audio, so keep answers concise, ideally a single sentence.
        Never read file names, source names, or chunk IDs out loud.
        If using the knowledge base:
        1. Always use the 'search' tool first.
        2. Always use the 'report_grounding' tool to cite sources used.
        3. If the answer isn't in the knowledge base, state that.
    """.strip()
    logger.info("RTMiddleTier initialized with system prompt.")

    # --- Attach WebSocket and Static Routes ---
    # RAG tools are NOT attached here; they are attached dynamically by update_rag_provider
    rtmt.attach_to_app(app, "/realtime")
    logger.info("Attached WebSocket handler to /realtime")

    # Add HTTP routes for RAG config and upload
    app.router.add_post("/rag-config", handle_rag_config)
    app.router.add_post("/upload", handle_upload)
    logger.info("Added HTTP routes: /rag-config (POST), /upload (POST)")


    # Serve static frontend files (assuming frontend build places files in backend/static)
    static_dir = BACKEND_DIR / 'static'
    if not static_dir.exists() or not (static_dir / 'index.html').exists():
        logger.warning(f"Static directory '{static_dir}' or index.html not found. Frontend may not be served.")
    else:
        logger.info(f"Serving static files from: {static_dir}")
        # Serve index.html at the root
        app.router.add_get('/', lambda _: web.FileResponse(static_dir / 'index.html'))
        # Serve other static files (JS, CSS, assets)
        app.router.add_static('/', path=static_dir, name='static', show_index=False) # Important: show_index=False

    return app

# --- Application Entry Point ---
if __name__ == "__main__":
    # Define an async main function to create the app instance
    async def main():
        # Setup logging levels based on environment or defaults
        # logging.getLogger('aiohttp.access').setLevel(logging.WARNING) # Example: reduce access log noise
        return await create_app()

    host = os.environ.get("BACKEND_HOST", "127.0.0.1") # Default to 127.0.0.1 for local dev
    port = int(os.environ.get("BACKEND_PORT", 8765)) # Use a distinct port

    logger.info(f"Starting application server on http://{host}:{port}")

    # Run the app using aiohttp's web runner
    # The 'main()' function returns the app instance created by create_app()
    web.run_app(main(), host=host, port=port)
