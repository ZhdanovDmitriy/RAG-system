CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents_short (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(768)
);

CREATE INDEX IF NOT EXISTS documents_short_embedding_idx
ON documents_short
USING ivfflat (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS documents_long (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(768)
);

CREATE INDEX IF NOT EXISTS documents_long_embedding_idx
ON documents_long
USING ivfflat (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS test_questions (
    id INTEGER PRIMARY KEY,
    question_text TEXT
);

CREATE TABLE IF NOT EXISTS test_answer_embeddings (
    id SERIAL PRIMARY KEY,
    question_id INTEGER REFERENCES test_questions(id),
    answer_text TEXT,
    embedding vector(768)
);

CREATE INDEX IF NOT EXISTS test_answer_embeddings_question_idx
ON test_answer_embeddings USING btree (question_id);