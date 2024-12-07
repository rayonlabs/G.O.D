-- migrate:up
ALTER TABLE tasks ADD COLUMN is_organic BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE tasks ADD COLUMN delay_times INTEGER NOT NULL DEFAULT 0;

-- migrate:down
ALTER TABLE tasks DROP COLUMN delay_times;
ALTER TABLE tasks DROP COLUMN is_organic;

