-- migrate:up
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'model_type_enum') THEN
        CREATE TYPE model_type_enum AS ENUM ('sdxl', 'flux');
    END IF;
END $$;
ALTER TABLE image_tasks
ADD COLUMN model_type model_type_enum DEFAULT 'sdxl';
UPDATE image_tasks
SET model_type = 'sdxl'
WHERE model_type IS NULL;
ALTER TABLE image_tasks
ALTER COLUMN model_type SET NOT NULL;

-- migrate:down
ALTER TABLE image_tasks
DROP COLUMN IF EXISTS model_type;
DROP TYPE IF EXISTS model_type_enum;
