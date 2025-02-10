-- migrate:up
ALTER TABLE tasks ADD COLUMN result_model_name TEXT;

-- migrate:down
ALTER TABLE tasks DROP COLUMN result_model_name;
