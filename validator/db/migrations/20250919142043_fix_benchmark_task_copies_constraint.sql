-- migrate:up
-- Drop the existing unique constraint
ALTER TABLE benchmark_task_copies 
DROP CONSTRAINT benchmark_task_copies_root_task_id_participant_hotkey_key;

-- Add new unique constraint that includes tournament_id
ALTER TABLE benchmark_task_copies 
ADD CONSTRAINT benchmark_task_copies_root_task_id_participant_hotkey_tournament_id_key 
UNIQUE (root_task_id, participant_hotkey, tournament_id);

-- migrate:down
-- Revert the changes
ALTER TABLE benchmark_task_copies 
DROP CONSTRAINT benchmark_task_copies_root_task_id_participant_hotkey_tournament_id_key;

ALTER TABLE benchmark_task_copies 
ADD CONSTRAINT benchmark_task_copies_root_task_id_participant_hotkey_key 
UNIQUE (root_task_id, participant_hotkey);
