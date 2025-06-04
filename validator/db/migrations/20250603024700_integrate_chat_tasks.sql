-- migrate:up
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_type t
        JOIN pg_enum e ON t.oid = e.enumtypid
        WHERE t.typname = 'tasktype' AND e.enumlabel = 'ChatTask'
    ) THEN
        ALTER TYPE tasktype ADD VALUE 'ChatTask';
    END IF;
END$$;

CREATE TABLE IF NOT EXISTS chat_tasks (
    task_id UUID PRIMARY KEY REFERENCES tasks(task_id) ON DELETE CASCADE,
    chat_template TEXT,
    chat_column TEXT,
    chat_role_field TEXT,
    chat_content_field TEXT,
    chat_user_reference TEXT,
    chat_assistant_reference TEXT,
    synthetic_data TEXT,
    file_format TEXT NOT NULL DEFAULT 'hf'
);

-- migrate:down

DROP TABLE IF EXISTS chat_tasks;

DELETE FROM tasks
  WHERE task_type = 'ChatTask';

ALTER TYPE tasktype RENAME TO tasktype_temp;
CREATE TYPE tasktype AS ENUM ('InstructTextTask', 'ImageTask', 'DpoTask', 'GrpoTask');

ALTER TABLE tasks
  ALTER COLUMN task_type TYPE VARCHAR;

ALTER TABLE tasks
  ALTER COLUMN task_type TYPE tasktype USING task_type::tasktype;

DROP TYPE tasktype_temp;
