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
    id SERIAL PRIMARY KEY,
    chat_template TEXT NOT NULL,
    chat_column TEXT NOT NULL,
    chat_role_field TEXT NOT NULL,
    chat_content_field TEXT NOT NULL,
    chat_user_reference TEXT NOT NULL,
    chat_assistant_reference TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- migrate:down

DROP TABLE IF EXISTS chat_tasks;

DELETE FROM tasks
  WHERE task_type = 'ChatTask';

ALTER TYPE tasktype RENAME TO tasktype_temp;
CREATE TYPE tasktype AS ENUM ('InstructTextTask', 'ImageTask', 'DpoTask');

ALTER TABLE tasks
  ALTER COLUMN task_type TYPE VARCHAR;

ALTER TABLE tasks
  ALTER COLUMN task_type TYPE tasktype USING task_type::tasktype;

DROP TYPE tasktype_temp;
