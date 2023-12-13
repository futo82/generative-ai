import datetime

from numpy import dot
from numpy.linalg import norm
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer
from sqlalchemy import Column
from sqlmodel import Field, SQLModel, Session, create_engine, select
from typing import Optional, List

class Concept(SQLModel, table=True):
    code: Optional[str] = Field(default=None, primary_key=True)
    parent: str
    definition: str
    display_name: str
    embedding: List[float] = Field(sa_column=Column(Vector(768)))
    date_uploaded: datetime.datetime = Field(default_factory=datetime.datetime.utcnow, nullable=False)

POSTGRES_URL = 'postgresql://postgres:mysecretpassword@localhost:5432/postgres' # TODO: Parameterize PostgreSQL URL
postgres_engine = create_engine(POSTGRES_URL, echo=False)
SQLModel.metadata.create_all(postgres_engine)

# Calculate the cosine similarity between 2 vectors
def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

# Perform a cosine distance select from the vector database
def retrieve_results(query_embedding, num_results):
    session = Session(postgres_engine)
    results = session.exec(select(Concept).order_by(Concept.embedding.cosine_distance(query_embedding)).limit(num_results))
    return results

def repl() -> None:
    try:
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2') # TODO: Parameterize the embedding model
        while True:
            try:
                query = input("Enter your query >> ")
                embeddings = model.encode([query])
                top_results = retrieve_results(embeddings[0], 10) # TODO: Parameterize the number of results
                print()
                print("Results: ")
                print("=============")
                for top_result in top_results:
                    print("Code: %s, Definition: %s, Similarity Score: %f" % (top_result.code, top_result.definition, cosine_similarity(top_result.embedding, embeddings[0])))
                    print()
                print()
            except Exception as e:
                print(f"Error: {e}")
    except KeyboardInterrupt as e:
        print("\nExiting...")

if __name__ == "__main__":
    print()
    print("Welcome to the NCIt CLI")
    print("crtl-c to quit")
    print()
    repl()