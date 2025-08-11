-- migrate:up

-- Benchmark root tasks table
-- This table stores task IDs that serve as templates for benchmarking
-- These are the original tasks that will be copied for each benchmark participant
CREATE TABLE IF NOT EXISTS benchmark_root_tasks (
    task_id UUID PRIMARY KEY REFERENCES tasks(task_id) ON DELETE CASCADE,
    task_type TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create index for efficient lookups by task type
CREATE INDEX IF NOT EXISTS idx_benchmark_root_tasks_task_type ON benchmark_root_tasks(task_type);

-- migrate:down
DROP TABLE IF EXISTS benchmark_root_tasks; 