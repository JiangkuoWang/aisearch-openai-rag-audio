import json
import logging
import os
import subprocess

from azure.core.exceptions import ResourceExistsError
from azure.identity import AzureDeveloperCliCredential
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    AzureOpenAIEmbeddingSkill,
    AzureOpenAIParameters,
    AzureOpenAIVectorizer,
    FieldMapping,
    HnswAlgorithmConfiguration,
    HnswParameters,
    IndexProjectionMode,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchIndexer,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    SearchIndexerDataSourceType,
    SearchIndexerIndexProjections,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters,
    SearchIndexerSkillset,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    SplitSkill,
    VectorSearch,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from rich.logging import RichHandler

# --- Configuration ---
# Determine the script's directory and the backend directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR) # Assumes script is in app/backend/scripts
# Path to the .env file in the backend directory
ENV_PATH = os.path.join(BACKEND_DIR, ".env")

# Setup logging FIRST
logging.basicConfig(level=logging.WARNING, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])
logger = logging.getLogger("voicerag")
logger.setLevel(logging.INFO)

def load_azd_env():
    """Get path to current azd env file and load file using python-dotenv"""
    try:
        result = subprocess.run("azd env list -o json", shell=True, capture_output=True, text=True, check=True)
        env_json = json.loads(result.stdout)
        env_file_path = None
        for entry in env_json:
            if entry["IsDefault"]:
                env_file_path = entry["DotEnvPath"]
        if not env_file_path:
            raise Exception("No default azd env file found in 'azd env list' output.")

        logger.info(f"Loading azd env from {env_file_path}")
        if not os.path.exists(env_file_path):
             logger.warning(f"azd default env file specified but not found at {env_file_path}. Attempting load from backend .env")
             env_file_path = ENV_PATH # Fallback to backend .env
        load_dotenv(env_file_path, override=True)
        logger.info(f"Successfully loaded environment variables from {env_file_path}")
        return True
    except FileNotFoundError:
        logger.error("'azd' command not found. Make sure Azure Developer CLI is installed and in your PATH.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running 'azd env list': {e}")
        logger.error(f"azd stdout: {e.stdout}")
        logger.error(f"azd stderr: {e.stderr}")
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON output from 'azd env list': {e}")
    except Exception as e:
        logger.error(f"Error loading azd env: {e}")

    # Fallback to loading .env directly from backend if azd load failed
    logger.warning(f"Could not load azd env. Attempting to load directly from {ENV_PATH}")
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH, override=True)
        logger.info(f"Successfully loaded environment variables from {ENV_PATH}")
        return True
    else:
        logger.error(f"Fallback .env file not found at {ENV_PATH}. Cannot load environment.")
        return False

def setup_index(azure_credential, index_name, azure_search_endpoint, azure_storage_connection_string, azure_storage_container, azure_openai_embedding_endpoint, azure_openai_embedding_deployment, azure_openai_embedding_model, azure_openai_embeddings_dimensions):
    index_client = SearchIndexClient(azure_search_endpoint, azure_credential)
    indexer_client = SearchIndexerClient(azure_search_endpoint, azure_credential)

    # --- Data Source Connection ---
    try:
        ds_connection = indexer_client.get_data_source_connection(index_name)
        logger.info(f"Data source connection '{index_name}' already exists.")
    except Exception: # Catches ResourceNotFoundError and potentially others
        logger.info(f"Creating data source connection: {index_name}")
        indexer_client.create_data_source_connection(
            data_source_connection=SearchIndexerDataSourceConnection(
                name=index_name,
                type=SearchIndexerDataSourceType.AZURE_BLOB,
                connection_string=azure_storage_connection_string,
                container=SearchIndexerDataContainer(name=azure_storage_container)))
        logger.info(f"Data source connection '{index_name}' created.")

    # --- Search Index ---
    try:
        index = index_client.get_index(index_name)
        logger.info(f"Index '{index_name}' already exists.")
        # TODO: Add logic here to update index if necessary based on schema changes
    except Exception:
        logger.info(f"Creating index: {index_name}")
        index_client.create_index(
            SearchIndex(
                name=index_name,
                fields=[
                    SimpleField(name="parent_id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
                    SearchableField(name="title", filterable=True, sortable=True),
                    SearchableField(name="chunk", filterable=True),
                    SearchField(
                        name="text_vector",
                        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        vector_search_dimensions=azure_openai_embeddings_dimensions, # Use passed dimension
                        vector_search_profile_name="default-hnsw-profile", # Updated name
                        hidden=False, # If you want to retrieve the vector
                        searchable=True # Must be true for vector search
                    )
                ],
                vector_search=VectorSearch(
                    algorithms=[
                        HnswAlgorithmConfiguration(
                            name="default-hnsw", # Updated name
                            parameters=HnswParameters(metric=VectorSearchAlgorithmMetric.COSINE))
                    ],
                    profiles=[
                        VectorSearchProfile(
                            name="default-hnsw-profile", # Updated name
                            algorithm_configuration_name="default-hnsw") # Updated name
                    ]
                    # Removed explicit vectorizer here - will be handled by skill
                ),
                semantic_search=SemanticSearch(
                    configurations=[
                        SemanticConfiguration(
                            name="default",
                            prioritized_fields=SemanticPrioritizedFields(title_field=SemanticField(field_name="title"), content_fields=[SemanticField(field_name="chunk")])
                        )
                    ]
                )
            )
        )
        logger.info(f"Index '{index_name}' created.")

    # --- Skillset ---
    try:
        skillset = indexer_client.get_skillset(index_name)
        logger.info(f"Skillset '{index_name}' already exists.")
        # TODO: Add update logic if necessary
    except Exception:
        logger.info(f"Creating skillset: {index_name}")
        indexer_client.create_skillset(
            skillset=SearchIndexerSkillset(
                name=index_name,
                description="Skillset for chunking and embedding documents",
                skills=[
                    SplitSkill(
                        name="split-skill", # Give skills unique names
                        description="Split content into pages",
                        text_split_mode="pages",
                        context="/document",
                        maximum_page_length=2000,
                        page_overlap_length=500,
                        inputs=[InputFieldMappingEntry(name="text", source="/document/content")],
                        outputs=[OutputFieldMappingEntry(name="textItems", target_name="pages")]),
                    AzureOpenAIEmbeddingSkill(
                        name="embedding-skill", # Give skills unique names
                        description="Generate embeddings using Azure OpenAI",
                        context="/document/pages/*",
                        resource_uri=azure_openai_embedding_endpoint,
                        # Use keyless auth if using managed identity, otherwise provide key
                        # api_key=os.environ.get("AZURE_OPENAI_API_KEY"), # Ensure this is the KEY for the embedding endpoint
                        deployment_id=azure_openai_embedding_deployment,
                        model_name=azure_openai_embedding_model, # Optional if deployment provides it
                        dimensions=azure_openai_embeddings_dimensions,
                        inputs=[InputFieldMappingEntry(name="text", source="/document/pages/*")],
                        outputs=[OutputFieldMappingEntry(name="embedding", target_name="vector")]) # Output target name is just 'vector'
                ],
                index_projections=SearchIndexerIndexProjections(
                    selectors=[
                        # Project chunks to the main index
                        SearchIndexerIndexProjectionSelector(
                            target_index_name=index_name,
                            parent_key_field_name="parent_id", # Field in index to store parent doc ID
                            source_context="/document/pages/*",
                            mappings=[
                                # Map the chunk content to the 'chunk' field
                                InputFieldMappingEntry(name="chunk", source="/document/pages/*"),
                                # Map the generated vector to the 'text_vector' field
                                InputFieldMappingEntry(name="text_vector", source="/document/pages/*/vector"),
                                # Map the original document file name to the 'title' field
                                InputFieldMappingEntry(name="title", source="/document/metadata_storage_name"),
                                # Use the doc ID as the chunk ID (parent_id in this case)
                                # InputFieldMappingEntry(name="chunk_id", source="/document/metadata_storage_path") # Map full path as ID if needed
                            ]
                        )
                    ],
                    parameters=SearchIndexerIndexProjectionsParameters(
                        projection_mode=IndexProjectionMode.SKIP_INDEXING_PARENT_DOCUMENTS
                    )
                )
            )
        )
        logger.info(f"Skillset '{index_name}' created.")

    # --- Indexer ---
    try:
        indexer = indexer_client.get_indexer(index_name)
        logger.info(f"Indexer '{index_name}' already exists.")
        # TODO: Add update logic if necessary
    except Exception:
        logger.info(f"Creating indexer: {index_name}")
        indexer_client.create_indexer(
            indexer=SearchIndexer(
                name=index_name,
                description="Indexer to process documents from blob storage",
                data_source_name=index_name, # Must match data source connection name
                skillset_name=index_name,    # Must match skillset name
                target_index_name=index_name, # Must match index name
                # Field mappings from the source document metadata to index fields
                field_mappings=[
                    # The parent document ID will be the blob path, map it to parent_id
                    FieldMapping(source_field_name="metadata_storage_path", target_field_name="parent_id"),
                    # Title is mapped within the skillset projection
                    # FieldMapping(source_field_name="metadata_storage_name", target_field_name="title")
                ],
                # Specify that the indexer should parse metadata and content
                parameters={
                    "configuration": {
                        "parsingMode": "default",
                        "dataToExtract": "contentAndMetadata"
                    }
                }
            )
        )
        logger.info(f"Indexer '{index_name}' created.")

def upload_documents(azure_credential, indexer_name, azure_search_endpoint, azure_storage_endpoint, azure_storage_container):
    indexer_client = SearchIndexerClient(azure_search_endpoint, azure_credential)
    logger.info(f"Connecting to Azure Storage: {azure_storage_endpoint}")
    # Upload the documents in DATA_SOURCE_DIR to the blob storage container
    data_source_dir = os.path.abspath(os.path.join(BACKEND_DIR, os.environ.get("DATA_SOURCE_DIR", "../../data")))
    if not os.path.isdir(data_source_dir):
        logger.error(f"Data source directory for upload not found: {data_source_dir}")
        return

    try:
        blob_service_client = BlobServiceClient(
            account_url=azure_storage_endpoint, credential=azure_credential,
            max_single_put_size=4 * 1024 * 1024 # 4 MiB
        )
        container_client = blob_service_client.get_container_client(azure_storage_container)
        if not container_client.exists():
            logger.info(f"Creating blob container: {azure_storage_container}")
            container_client.create_container()

        logger.info(f"Checking for existing blobs in container '{azure_storage_container}'...")
        existing_blobs = {blob.name for blob in container_client.list_blobs()}
        logger.info(f"Found {len(existing_blobs)} existing blobs.")

        logger.info(f"Scanning local data directory for upload: {data_source_dir}")
        files_uploaded = 0
        files_skipped = 0
        # Open each file in data_source_dir
        for file in os.scandir(data_source_dir):
            if file.is_file() and not file.name.startswith('.'):
                filename = file.name
                filepath = file.path
                # Check if blob already exists
                if filename in existing_blobs:
                    # logger.debug(f"Blob already exists, skipping file: {filename}")
                    files_skipped += 1
                else:
                    logger.info(f"  Uploading blob for file: {filename}")
                    try:
                        with open(filepath, "rb") as data:
                            container_client.upload_blob(name=filename, data=data, overwrite=True)
                        files_uploaded += 1
                    except Exception as upload_error:
                        logger.error(f"  Error uploading file {filename}: {upload_error}")

        logger.info(f"Finished upload scan. Uploaded: {files_uploaded}, Skipped (already exist): {files_skipped}")

    except Exception as storage_error:
        logger.error(f"Error interacting with Azure Storage: {storage_error}")
        return # Don't try to run indexer if upload failed

    # Start the indexer if files were uploaded or if forced
    if files_uploaded > 0:
        logger.info("New files were uploaded, running indexer...")
        try:
            indexer_client.run_indexer(indexer_name)
            logger.info(f"Indexer '{indexer_name}' started. Check the Azure Portal for status.")
        except ResourceExistsError:
            logger.warning(f"Indexer '{indexer_name}' is already running. New blobs will be picked up.")
        except Exception as indexer_error:
            logger.error(f"Error running indexer '{indexer_name}': {indexer_error}")
    else:
        logger.info("No new files uploaded, indexer run not explicitly triggered.")

if __name__ == "__main__":

    if not load_azd_env():
        logger.critical("Failed to load environment variables. Exiting.")
        exit(1)

    logger.info("Checking if Azure AI Search setup is required...")
    if os.environ.get("AZURE_SEARCH_REUSE_EXISTING", "false").lower() == "true":
        logger.info("AZURE_SEARCH_REUSE_EXISTING is true. Skipping Azure AI Search setup.")
        exit()
    else:
        logger.info("Setting up Azure AI Search index and integrated vectorization...")

    try:
        # Retrieve required environment variables
        AZURE_SEARCH_INDEX = os.environ["AZURE_SEARCH_INDEX"]
        AZURE_OPENAI_EMBEDDING_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
        AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
        # Get embedding model from env, default is optional here if deployment implies model
        AZURE_OPENAI_EMBEDDING_MODEL = os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL")
        # Get dimensions - THIS IS CRITICAL and depends on the model used.
        # Get dimension for the chosen model
        MODEL_DIMENSIONS = {
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536,
            "text-embedding-3-large": 3072,
        }
        # Use the model name specified in env (could be ada, 3-small, 3-large etc)
        EMBEDDING_MODEL_NAME_FOR_DIM = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        EMBEDDINGS_DIMENSIONS = MODEL_DIMENSIONS.get(EMBEDDING_MODEL_NAME_FOR_DIM)
        if EMBEDDINGS_DIMENSIONS is None:
             logger.warning(f"Cannot determine embedding dimension for model '{EMBEDDING_MODEL_NAME_FOR_DIM}' from environment variable OPENAI_EMBEDDING_MODEL. Using default 1536.")
             EMBEDDINGS_DIMENSIONS = 1536 # Default or raise error

        AZURE_SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
        AZURE_STORAGE_ENDPOINT = os.environ["AZURE_STORAGE_ENDPOINT"]
        AZURE_STORAGE_CONNECTION_STRING = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        AZURE_STORAGE_CONTAINER = os.environ["AZURE_STORAGE_CONTAINER"]
        AZURE_TENANT_ID = os.environ["AZURE_TENANT_ID"]

        # Validate endpoints format
        if not AZURE_SEARCH_ENDPOINT.startswith("https://") or not AZURE_SEARCH_ENDPOINT.endswith(".search.windows.net"):
            raise ValueError(f"Invalid AZURE_SEARCH_ENDPOINT format: {AZURE_SEARCH_ENDPOINT}")
        if not AZURE_STORAGE_ENDPOINT.startswith("https://") or not AZURE_STORAGE_ENDPOINT.endswith(".blob.core.windows.net"):
             raise ValueError(f"Invalid AZURE_STORAGE_ENDPOINT format: {AZURE_STORAGE_ENDPOINT}")

        # Get credentials
        logger.info("Authenticating using AzureDeveloperCliCredential...")
        azure_credential = AzureDeveloperCliCredential(tenant_id=AZURE_TENANT_ID, process_timeout=60)
        # Trigger auth flow early
        azure_credential.get_token("https://search.azure.com/.default")
        logger.info("Authentication successful.")

        # Setup Index, Skillset, Indexer
        setup_index(azure_credential,
            index_name=AZURE_SEARCH_INDEX,
            azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
            azure_storage_connection_string=AZURE_STORAGE_CONNECTION_STRING,
            azure_storage_container=AZURE_STORAGE_CONTAINER,
            azure_openai_embedding_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
            azure_openai_embedding_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            azure_openai_embedding_model=AZURE_OPENAI_EMBEDDING_MODEL,
            azure_openai_embeddings_dimensions=EMBEDDINGS_DIMENSIONS)

        # Upload documents and run indexer
        upload_documents(azure_credential,
            indexer_name=AZURE_SEARCH_INDEX,
            azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
            azure_storage_endpoint=AZURE_STORAGE_ENDPOINT,
            azure_storage_container=AZURE_STORAGE_CONTAINER)

        logger.info("Azure AI Search setup script completed successfully.")

    except KeyError as e:
        logger.critical(f"Missing required environment variable: {e}. Please ensure your .env file is correctly populated.")
        exit(1)
    except ValueError as e:
         logger.critical(f"Configuration error: {e}")
         exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during Azure AI Search setup: {e}", exc_info=True)
        exit(1) 