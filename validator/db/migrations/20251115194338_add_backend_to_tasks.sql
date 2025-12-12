-- migrate:up

ALTER TABLE tasks
ADD COLUMN IF NOT EXISTS backend TEXT;

-- Create index on backend for efficient filtering
CREATE INDEX IF NOT EXISTS idx_tasks_backend ON tasks(backend);

-- migrate:down

-- Drop the index first
DROP INDEX IF EXISTS idx_tasks_backend;

-- Then drop the column
ALTER TABLE tasks DROP COLUMN IF EXISTS backend;

