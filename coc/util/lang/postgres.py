from langgraph.store.postgres import PostgresStore
import asyncio

conn_string = "postgresql://jkp: @localhost:5432/une"

with PostgresStore.from_conn_string(conn_string) as store:
    store.setup()  # Creates necessary tables

from langchain.embeddings import init_embeddings
from langgraph.store.postgres import AsyncPostgresStore

conn_string = "postgresql://jkp: @localhost:5432/une"

with PostgresStore.from_conn_string(
    conn_string,
    index={
        "dims": 1536,
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "fields": ["text"],  # specify which fields to embed. Default is the whole serialized value
        "openai_api_key": "sk-aDjzEoivIbVx4o9xLIEUDrpRaNTDOOhW1rTPhCsGsdjTa3Or",
        "openai_api_base": "https://chatapi.littlewheat.com/v1",
    }
) as store:
    store.setup()  # Run migrations. Done once

    # Store documents
    store.put(("docs",), "doc1", {"text": "Python tutorial"})
    store.put(("docs",), "doc2", {"text": "TypeScript guide"})
    store.put(("docs",), "doc3", {"text": "Other guide"}, index=False)  # don't index

    # Search by similarity
    results = store.search(("docs",), query="programming guides", limit=2)

    print(results)