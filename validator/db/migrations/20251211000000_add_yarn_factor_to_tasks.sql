-- migrate:up

ALTER TABLE tasks
ADD COLUMN IF NOT EXISTS yarn_factor INTEGER;

-- migrate:down

-- Drop the column
ALTER TABLE tasks DROP COLUMN IF EXISTS yarn_factor;
