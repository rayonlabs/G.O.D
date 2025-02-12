-- migrate:up

CREATE EXTENSION IF NOT EXISTS pg_trgm;
ALTER TABLE tasks ADD COLUMN result_model_name_lower TEXT GENERATED ALWAYS AS (LOWER(result_model_name)) STORED;
CREATE INDEX idx_tasks_result_model_name_lower ON tasks USING gin (result_model_name_lower gin_trgm_ops);

-- migrate:down

DROP INDEX IF EXISTS idx_tasks_result_model_name_lower;
DROP EXTENSION IF EXISTS pg_trgm;
ALTER TABLE tasks DROP COLUMN result_model_name_lower;