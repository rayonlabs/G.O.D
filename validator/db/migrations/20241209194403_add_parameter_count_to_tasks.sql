-- migrate:up
ALTER TABLE tasks ADD COLUMN parameter_count BIGINT;

-- migrate:down
ALTER TABLE tasks DROP COLUMN parameter_count;
