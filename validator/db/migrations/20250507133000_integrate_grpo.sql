-- migrate:up

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_enum
        WHERE enumlabel = 'GrpoTask'
        AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'tasktype')
    ) THEN
        ALTER TYPE tasktype ADD VALUE 'GrpoTask';
    END IF;
END $$;

-- Create the GRPO tasks table
CREATE TABLE IF NOT EXISTS grpo_tasks (
    task_id UUID PRIMARY KEY REFERENCES tasks(task_id) ON DELETE CASCADE,
    field_prompt TEXT NOT NULL,
    synthetic_data TEXT,
    file_format TEXT NOT NULL DEFAULT 'hf'
);

-- Create the reward functions table (reusable functions)
CREATE TABLE IF NOT EXISTS reward_functions (
    reward_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reward_func TEXT NOT NULL,
    func_hash CHAR(64) NOT NULL,  -- SHA256 hash for uniqueness
    is_generic BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create the junction table for tasks and functions
CREATE TABLE IF NOT EXISTS grpo_task_functions (
    task_id UUID NOT NULL REFERENCES grpo_tasks(task_id) ON DELETE CASCADE,
    reward_id UUID NOT NULL REFERENCES reward_functions(reward_id) ON DELETE RESTRICT,
    reward_weight FLOAT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (task_id, reward_id),
    CONSTRAINT valid_reward_weight CHECK (reward_weight >= 0)
);

-- Create indexes for better query performance
CREATE UNIQUE INDEX IF NOT EXISTS idx_reward_functions_hash ON reward_functions(func_hash);
CREATE INDEX IF NOT EXISTS idx_grpo_task_functions_task_id ON grpo_task_functions(task_id);
CREATE INDEX IF NOT EXISTS idx_grpo_task_functions_reward_id ON grpo_task_functions(reward_id);

-- migrate:down
DROP TABLE IF EXISTS grpo_task_functions;
DROP TABLE IF EXISTS reward_functions;
DROP TABLE IF EXISTS grpo_tasks;

DELETE FROM tasks
  WHERE task_type = 'GrpoTask';

ALTER TYPE tasktype RENAME TO tasktype_temp;
CREATE TYPE tasktype AS ENUM ('InstructTextTask', 'ImageTask', 'DpoTask');

ALTER TABLE tasks
  ALTER COLUMN task_type TYPE VARCHAR;

ALTER TABLE tasks
  ALTER COLUMN task_type TYPE tasktype USING task_type::tasktype;

DROP TYPE tasktype_temp;
