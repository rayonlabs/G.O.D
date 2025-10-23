-- migrate:up

-- Add training_repo and training_commit_hash columns to tournament_task_hotkey_trainings table
ALTER TABLE tournament_task_hotkey_trainings
ADD COLUMN IF NOT EXISTS training_repo TEXT,
ADD COLUMN IF NOT EXISTS training_commit_hash TEXT;

-- Create indexes for the new columns for efficient lookups
CREATE INDEX IF NOT EXISTS idx_tournament_task_hotkey_trainings_training_repo
ON tournament_task_hotkey_trainings(training_repo);

-- migrate:down

-- Drop the indexes first
DROP INDEX IF EXISTS idx_tournament_task_hotkey_trainings_training_repo;

-- Then drop the columns
ALTER TABLE tournament_task_hotkey_trainings
DROP COLUMN IF EXISTS training_repo,
DROP COLUMN IF EXISTS training_commit_hash;