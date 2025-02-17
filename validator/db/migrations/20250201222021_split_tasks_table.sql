-- migrate:up
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'tasktype') THEN
        CREATE TYPE tasktype AS ENUM ('TextTask', 'ImageTask');
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS text_tasks (
    task_id UUID PRIMARY KEY REFERENCES tasks (task_id) ON DELETE CASCADE,
    field_system TEXT,
    field_instruction TEXT,
    field_input TEXT,
    field_output TEXT,
    synthetic_data TEXT,
    format TEXT,
    no_input_format TEXT,
    system_format TEXT,
    file_format TEXT NOT NULL DEFAULT 'hf'
);

CREATE TABLE IF NOT EXISTS image_tasks (
    task_id UUID PRIMARY KEY REFERENCES tasks (task_id) ON DELETE CASCADE,
    model_filename TEXT
);

CREATE INDEX IF NOT EXISTS idx_text_task_id ON text_tasks (task_id);

CREATE INDEX IF NOT EXISTS idx_image_task_id ON image_tasks (task_id);

ALTER TABLE tasks ADD COLUMN task_type tasktype;
ALTER TABLE tasks RENAME COLUMN ds_id TO ds;


UPDATE tasks SET task_type = 'TextTask';


ALTER TABLE tasks ALTER COLUMN task_type SET NOT NULL;

INSERT INTO text_tasks (
    task_id,
    field_system,
    field_instruction,
    field_input,
    field_output,
    synthetic_data,
    format,
    no_input_format,
    system_format,
    file_format
)
SELECT
    task_id,
    field_system,
    field_instruction,
    field_input,
    field_output,
    synthetic_data,
    format,
    no_input_format,
    system_format,
    file_format
FROM tasks;

ALTER TABLE tasks
DROP COLUMN field_system,
DROP COLUMN field_instruction,
DROP COLUMN field_input,
DROP COLUMN field_output,
DROP COLUMN synthetic_data,
DROP COLUMN format,
DROP COLUMN no_input_format,
DROP COLUMN system_format,
DROP COLUMN file_format;


-- migrate:down
DROP TABLE IF EXISTS text_tasks;
DROP TABLE IF EXISTS image_tasks;

DROP INDEX IF EXISTS idx_text_task_id;
DROP INDEX IF EXISTS idx_image_task_id;

ALTER TABLE tasks
RENAME COLUMN ds TO ds_id;

ALTER TABLE tasks
ADD COLUMN field_system TEXT,
ADD COLUMN field_instruction TEXT,
ADD COLUMN field_input TEXT,
ADD COLUMN field_output TEXT,
ADD COLUMN synthetic_data TEXT,
ADD COLUMN format TEXT,
ADD COLUMN no_input_format TEXT,
ADD COLUMN system_format TEXT,
ADD COLUMN file_format TEXT NOT NULL DEFAULT 'hf';

ALTER TABLE tasks DROP COLUMN task_type;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_type WHERE typname = 'tasktype') THEN
        DROP TYPE tasktype;
    END IF;
END $$;
