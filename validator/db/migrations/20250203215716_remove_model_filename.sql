-- migrate:up
ALTER TABLE image_tasks DROP COLUMN model_filename;

-- migrate:down
ALTER TABLE image_tasks ADD COLUMN model_filename TEXT;
