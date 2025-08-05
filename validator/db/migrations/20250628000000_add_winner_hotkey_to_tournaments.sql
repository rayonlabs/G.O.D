-- migrate:up

ALTER TABLE tournaments 
ADD COLUMN winner_hotkey TEXT;

CREATE INDEX IF NOT EXISTS idx_tournaments_winner_hotkey ON tournaments(winner_hotkey);

-- migrate:down

DROP INDEX IF EXISTS idx_tournaments_winner_hotkey;

ALTER TABLE tournaments 
DROP COLUMN IF EXISTS winner_hotkey; 