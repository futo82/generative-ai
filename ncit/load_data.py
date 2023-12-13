import csv
import datetime

from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer
from sqlalchemy import Column
from sqlmodel import Field, SQLModel, Session, create_engine
from typing import Optional, List

def split(a_list, chunk_size):
    for i in range(0, len(a_list), chunk_size):
        yield a_list[i:i + chunk_size]

class Concept(SQLModel, table=True):
    code: Optional[str] = Field(default=None, primary_key=True)
    parent: str
    definition: str
    display_name: str
    embedding: List[float] = Field(sa_column=Column(Vector(768))) # TODO: Parameterize the vector size
    date_uploaded: datetime.datetime = Field(default_factory=datetime.datetime.utcnow, nullable=False)

'''
The Thesaurus.txt flat file is in tab-delimited format.  Included in this format
are all the terms associated with NCI Thesaurus concepts (names and synonyms), a text
definition of the concept (if one is present), and stated parent-child relations, sufficient
to reconstruct the hierarchy.  The fields are:

code <tab> concept IRI <tab> parents <tab> synonyms <tab> definition <tab> display name <tab> concept status <tab> semantic type <tab> concept in subset <EOL>
'''
concepts = []
with open('Thesaurus.txt', newline = '') as f:
    print('Reading file ...')
    concept_reader = csv.reader(f, delimiter='\t')
    for entry in concept_reader:
    	concepts.append(Concept(
            code = entry[0],
            parent = entry[2],
            definition = entry[4],
            display_name = entry[5]
        ))

concept_chunks = list(split(concepts, 100)) # TODO: Parameterize the chunk size
print("Created %i chunks." % len(concept_chunks))

POSTGRES_URL = 'postgresql://postgres:mysecretpassword@localhost:5432/postgres' # TODO: Parameterize PostgreSQL URL
postgres_engine = create_engine(POSTGRES_URL, echo=True)
SQLModel.metadata.create_all(postgres_engine)
session = Session(postgres_engine)

chunk_counter = 0
num_chunks = len(concept_chunks)
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2') # TODO: Parameterize the embedding model
for concept_chunk in concept_chunks:
    print("Processing chunk %i ..." % chunk_counter)
    sentences = [c.definition for c in concept_chunk]
    embeddings = model.encode(sentences)
    for i in range(len(concept_chunk)):
        concept_chunk[i].embedding = embeddings[i]
        session.add(concept_chunk[i])
    session.commit()
    chunk_counter = chunk_counter + 1
