-- migrate:up
ALTER TYPE tasktype RENAME TO tasktype_old;
CREATE TYPE tasktype AS ENUM ('InstructTextTask', 'ImageTask', 'DpoTask');

-- Convert existing data to the new enum type
ALTER TABLE tasks
  ALTER COLUMN task_type TYPE VARCHAR;

UPDATE tasks
  SET task_type = 'InstructTextTask'
  WHERE task_type = 'TextTask';

ALTER TABLE text_tasks RENAME TO instruct_text_tasks;
ALTER INDEX idx_text_task_id RENAME TO idx_instruct_text_task_id;

ALTER TABLE tasks
  ALTER COLUMN task_type TYPE tasktype USING task_type::tasktype;

DROP TYPE tasktype_old;

CREATE TABLE IF NOT EXISTS dpo_tasks (
    task_id UUID PRIMARY KEY REFERENCES tasks (task_id) ON DELETE CASCADE,
    field_prompt TEXT,
    field_system TEXT,
    field_chosen TEXT,
    field_rejected TEXT,
    prompt_format TEXT,
    chosen_format TEXT,
    rejected_format TEXT,
    synthetic_data TEXT,
    file_format TEXT NOT NULL DEFAULT 'hf'
);

CREATE INDEX IF NOT EXISTS idx_dpo_task_id ON dpo_tasks (task_id);

-- migrate:down
DROP TABLE IF EXISTS dpo_tasks;
DROP INDEX IF EXISTS idx_dpo_task_id;

ALTER TABLE instruct_text_tasks RENAME TO text_tasks;
ALTER INDEX idx_instruct_text_task_id RENAME TO idx_text_task_id;

DELETE FROM tasks
  WHERE task_type = 'DpoTask';

-- Revert tasktype enum changes
ALTER TYPE tasktype RENAME TO tasktype_temp;
CREATE TYPE tasktype AS ENUM ('TextTask', 'ImageTask');

-- Convert data back
ALTER TABLE tasks
  ALTER COLUMN task_type TYPE VARCHAR;

UPDATE tasks
  SET task_type = 'TextTask'
  WHERE task_type = 'InstructTextTask';

ALTER TABLE tasks
  ALTER COLUMN task_type TYPE tasktype USING task_type::tasktype;

DROP TYPE tasktype_temp;
