-- migrate:up
ALTER TABLE image_tasks
ADD COLUMN is_flux_model BOOLEAN NOT NULL DEFAULT FALSE;

-- migrate:down
ALTER TABLE image_tasks
DROP COLUMN is_flux_model;
