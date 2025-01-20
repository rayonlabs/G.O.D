-- migrate:up
ALTER TABLE tasks
ADD COLUMN model_params_count BIGINT DEFAULT 0;

-- migrate:down
ALTER TABLE tasks
DROP COLUMN model_params_count;
