-- migrate:up

-- Add trainer_ip column to tournament_task_hotkey_trainings table
ALTER TABLE tournament_task_hotkey_trainings
ADD COLUMN IF NOT EXISTS trainer_ip TEXT;

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_tournament_task_hotkey_trainings_trainer_ip 
ON tournament_task_hotkey_trainings(trainer_ip);

-- migrate:down

DROP INDEX IF EXISTS idx_tournament_task_hotkey_trainings_trainer_ip;
ALTER TABLE tournament_task_hotkey_trainings DROP COLUMN IF EXISTS trainer_ip;

