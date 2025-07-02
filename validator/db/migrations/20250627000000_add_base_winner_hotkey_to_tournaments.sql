-- migrate:up

ALTER TABLE tournaments 
ADD COLUMN base_winner_hotkey TEXT;

CREATE INDEX IF NOT EXISTS idx_tournaments_base_winner_hotkey ON tournaments(base_winner_hotkey);

-- migrate:down

DROP INDEX IF EXISTS idx_tournaments_base_winner_hotkey;

ALTER TABLE tournaments 
DROP COLUMN IF EXISTS base_winner_hotkey; 