-- migrate:up


-- Add priority column to tournament_task_hotkey_trainings table
-- Priority 1: Regular tournament tasks (default)
-- Priority 2: Benchmark tasks

ALTER TABLE tournament_task_hotkey_trainings 
ADD COLUMN IF NOT EXISTS priority INTEGER DEFAULT 1;

-- Create index on priority for efficient sorting
CREATE INDEX IF NOT EXISTS idx_tournament_task_hotkey_trainings_priority 
ON tournament_task_hotkey_trainings(priority);

-- migrate:down

ALTER TABLE tournament_task_hotkey_trainings DROP COLUMN IF EXISTS priority;

DROP INDEX IF EXISTS idx_tournament_task_hotkey_trainings_priority;