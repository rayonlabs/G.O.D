-- migrate:up

-- Add training status fields to tournament_pairs table
ALTER TABLE tournament_pairs
ADD COLUMN is_hotkey1_trained BOOLEAN DEFAULT FALSE,
ADD COLUMN is_hotkey2_trained BOOLEAN DEFAULT FALSE;

-- Add training status field to tournament_group_members table
ALTER TABLE tournament_group_members
ADD COLUMN is_hotkey_trained BOOLEAN DEFAULT FALSE;

-- migrate:down

-- Remove training status fields from tournament_pairs table
ALTER TABLE tournament_pairs
DROP COLUMN IF EXISTS is_hotkey1_trained,
DROP COLUMN IF EXISTS is_hotkey2_trained;

-- Remove training status field from tournament_group_members table
ALTER TABLE tournament_group_members
DROP COLUMN IF EXISTS is_hotkey_trained;
