import argparse
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
    embedding: List[float] = Field(sa_column=Column(Vector(768)))
    date_uploaded: datetime.datetime = Field(default_factory=datetime.datetime.utcnow, nullable=False)

parser = argparse.ArgumentParser(
    prog='load_data.py',
    description='This program reads a NCI Thesaurus text file, converts the definition into a vector embedding, and stores the embedding into a PostgreSQL database with the pgvector extension.')
parser.add_argument('--postgres_url', help='The url to connect to the PostgreSQL database.')
parser.add_argument('--embedding_model', help='The embedding model to use.')
parser.add_argument('--chunk_size', type=int, help='The number of concepts to insert into the database at a time.')
parser.add_argument('--thesaurus_filepath', type=int, help='The filepath to the thesaurus text file.')
args = parser.parse_args()

POSTGRES_URL = 'postgresql://postgres:mysecretpassword@localhost:5432/postgres' 
if args.postgres_url:
    print("Setting POSTGRES_URL to %s ..." % args.postgres_url)
    POSTGRES_URL = args.postgres_url

EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
if args.embedding_model:
    print("Setting EMBEDDING_MODEL to %s ..." % args.embedding_model)
    EMBEDDING_MODEL = args.embedding_model

CHUNK_SIZE = 100
if args.chunk_size:
    print("Setting CHUNK_SIZE to %i ..." % args.chunk_size)
    CHUNK_SIZE = args.chunk_size

THESAURUS_FILEPATH = 'Thesaurus.txt'
if args.thesaurus_filepath:
    print("Setting THESAURUS_FILEPATH to %s ..." % args.thesaurus_filepath)
    THESAURUS_FILEPATH = args.thesaurus_filepath

'''
The Thesaurus.txt flat file is in tab-delimited format.  Included in this format
are all the terms associated with NCI Thesaurus concepts (names and synonyms), a text
definition of the concept (if one is present), and stated parent-child relations, sufficient
to reconstruct the hierarchy.  The fields are:

code <tab> concept IRI <tab> parents <tab> synonyms <tab> definition <tab> display name <tab> concept status <tab> semantic type <tab> concept in subset <EOL>
'''
concepts = []
with open(THESAURUS_FILEPATH, newline = '') as f:
    print('Reading file ...')
    concept_reader = csv.reader(f, delimiter='\t')
    for entry in concept_reader:
    	concepts.append(Concept(
            code = entry[0],
            parent = entry[2],
            definition = entry[4],
            display_name = entry[5]
        ))

concept_chunks = list(split(concepts, CHUNK_SIZE))
print("Created %i chunks." % len(concept_chunks))

postgres_engine = create_engine(POSTGRES_URL, echo=True)
SQLModel.metadata.create_all(postgres_engine)
session = Session(postgres_engine)

chunk_counter = 0
num_chunks = len(concept_chunks)
model = SentenceTransformer(EMBEDDING_MODEL)
for concept_chunk in concept_chunks:
    print("Processing chunk %i ..." % chunk_counter)
    sentences = [c.definition for c in concept_chunk]
    embeddings = model.encode(sentences)
    for i in range(len(concept_chunk)):
        concept_chunk[i].embedding = embeddings[i]
        session.add(concept_chunk[i])
    session.commit()
    chunk_counter = chunk_counter + 1
