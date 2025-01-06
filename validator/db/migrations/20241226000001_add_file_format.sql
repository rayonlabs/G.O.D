-- migrate:up
ALTER TABLE tasks ADD COLUMN file_format TEXT NOT NULL DEFAULT 'hf';
ALTER TABLE tasks ALTER COLUMN file_format DROP DEFAULT;

-- migrate:down
ALTER TABLE tasks DROP COLUMN file_format;
