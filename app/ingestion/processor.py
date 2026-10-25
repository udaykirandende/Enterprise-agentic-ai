import os
import uuid
import sys
import json
import logfire

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text
from app.ingestion.loaders.office import parse_office
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.chunking.splitter import chunk_text
from app.services.retrival.embeddings import embed_texts, get_embedding_dim


logfire.configure(service_name="UDAY-CHATBOT")

processed_data_dir = "processed_data"

Qdrant_Client = QdrantClient(
    api_key=settings.QDRANT_API_KEY,
    url=settings.QDRANT_URL
)


def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    folder = os.path.join(processed_data_dir, source_type)
    os.makedirs(folder, exist_ok=True)

    dest = os.path.join(folder, filename)

    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return dest


def process_file(file_path: str, filename: str, source_type: str) -> str:
    with logfire.span(
        "Processing file",
        file_path=file_path,
        source_type=source_type
    ):
        try:
            # 1. Extract text based on file extension
            ext = filename.lower().rsplit(".", 1)[-1]

            if ext == "pdf":
                full_text = parse_pdf(file_path)

            elif ext in ("html", "htm"):
                full_text = parse_html(file_path)

            elif ext == "txt":
                full_text = parse_text(file_path)

            elif ext in ("docx", "pptx"):
                full_text = parse_office(file_path)

            else:
                logfire.warning(
                    f"Skipping unsupported file type: {filename}"
                )
                return

            if not full_text or not full_text.strip():
                logfire.warning(
                    f"No text extracted from {filename} — skipping."
                )
                return

            # 2. Chunk text
            chunks = chunk_text(full_text)

            if not chunks:
                logfire.warning(
                    f"No chunks generated from {filename} — skipping."
                )
                return

            logfire.info(
                f"Generated {len(chunks)} chunks from {filename}."
            )

            # 3. Save processed metadata locally
            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "chunks": chunks,
            }

            local_path = save_processed_locally(
                processed_data,
                source_type,
                filename
            )

            logfire.info(
                f"Saved processed data → {local_path}"
            )

            # 4. Embed and index in Qdrant
            with logfire.span("Vectorizing & Indexing"):

                embeddings = embed_texts(chunks)

                
                points = [
                     models.PointStruct(
                     id=str(uuid.uuid5(
                     uuid.NAMESPACE_URL,
                    f"{filename}:{source_type}:{index}:{chunk}"
                     )),
                 vector=vector,
                 payload={
                 "text": chunk,
                "source": filename,
                 "source_type": source_type,
                },
                )
                 for index, (chunk, vector) in enumerate(zip(chunks, embeddings))
                ]

                # Upload to Qdrant in smaller batches
                batch_size = 25
                total_points = len(points)

                for i in range(0, total_points, batch_size):
                    batch = points[i:i + batch_size]

                    Qdrant_Client.upsert(
                        collection_name=settings.QDRANT_COLLECTION,
                        points=batch,
                    )

                    batch_number = (i // batch_size) + 1
                    total_batches = (
                        total_points + batch_size - 1
                    ) // batch_size

                    logfire.info(
                        f"Uploaded batch {batch_number}/{total_batches} "
                        f"({len(batch)} points) to Qdrant."
                    )

                logfire.info(
                    f"Indexed {total_points} points to Qdrant "
                    f"from {filename}."
                )

        except Exception as e:
            logfire.error(
                f"Failed to process {filename}: {e}"
            )


def process_directory(dir_path: str, source_type: str):
    """Process every file in a directory."""

    with logfire.span(
        "Scanning Directory",
        path=dir_path,
        source=source_type
    ):
        files = [
            f
            for f in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, f))
        ]

        logfire.info(
            f"Found {len(files)} files in {dir_path}."
        )

        for filename in files:
            process_file(
                os.path.join(dir_path, filename),
                filename,
                source_type
            )


def run_universal_ingestion(
    base_dir: str,
    explicit_source_type: str = None,
    wipe: bool = False
):
    """
    Scan base_dir, map sub-folders to source types,
    and ingest all documents.

    Pass --wipe to drop and recreate the Qdrant collection
    before ingestion.
    """

    with logfire.span(
        "Universal Ingestion Started",
        base_directory=base_dir
    ):

        # 1. Wipe collection if requested
        if wipe:
            with logfire.span("Wiping Collection"):

                if Qdrant_Client.collection_exists(
                    settings.QDRANT_COLLECTION
                ):
                    Qdrant_Client.delete_collection(
                        settings.QDRANT_COLLECTION
                    )

                    logfire.info(
                        f"Collection '{settings.QDRANT_COLLECTION}' deleted."
                    )

        # 2. Create collection if it doesn't exist
        if not Qdrant_Client.collection_exists(
            settings.QDRANT_COLLECTION
        ):

            dim = get_embedding_dim()

            Qdrant_Client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )

            logfire.info(
                f"Created collection '{settings.QDRANT_COLLECTION}' "
                f"({dim}-dim, Cosine)."
            )

        # 3. Find subdirectories
        subdirs = [
            d
            for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]

        # 4. Process directory
        if not subdirs:

            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(
                    os.path.normpath(base_dir)
                ).lower()

                source_type = (
                    "true"
                    if "true" in base_name
                    else "noisy"
                    if "noisy" in base_name
                    else "general"
                )

            logfire.info(
                f"No sub-folders found — processing "
                f"'{base_dir}' as '{source_type}'."
            )

            process_directory(
                base_dir,
                source_type
            )

        else:

            for subdir in subdirs:

                source_type = (
                    "true"
                    if "true" in subdir.lower()
                    else "noisy"
                    if "noisy" in subdir.lower()
                    else subdir
                )

                process_directory(
                    os.path.join(base_dir, subdir),
                    source_type
                )


if __name__ == "__main__":

    # Usage:
    # python -m app.ingestion.processor data
    # python -m app.ingestion.processor data --wipe
    # python -m app.ingestion.processor data/true_data true

    wipe_requested = "--wipe" in sys.argv

    clean_args = [
        a for a in sys.argv
        if a != "--wipe"
    ]

    target_dir = (
        clean_args[1]
        if len(clean_args) > 1
        else "data"
    )

    explicit_type = (
        clean_args[2]
        if len(clean_args) > 2
        else None
    )

    if not os.path.exists(target_dir):
        print(
            f"Error: path '{target_dir}' does not exist."
        )
        sys.exit(1)

    run_universal_ingestion(
        target_dir,
        explicit_source_type=explicit_type,
        wipe=wipe_requested
    )

    logfire.info(
        "Ingestion job completed."
    )