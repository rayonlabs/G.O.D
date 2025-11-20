-- migrate:up

ALTER TABLE tournament_task_hotkey_trainings
ADD COLUMN IF NOT EXISTS dstack_runname TEXT;

-- Create index on dstack_runname for efficient lookups
CREATE INDEX IF NOT EXISTS idx_tournament_task_hotkey_trainings_dstack_runname
ON tournament_task_hotkey_trainings(dstack_runname);

-- migrate:down

-- Drop the index first
DROP INDEX IF EXISTS idx_tournament_task_hotkey_trainings_dstack_runname;

-- Then drop the column
ALTER TABLE tournament_task_hotkey_trainings DROP COLUMN IF EXISTS dstack_runname;

