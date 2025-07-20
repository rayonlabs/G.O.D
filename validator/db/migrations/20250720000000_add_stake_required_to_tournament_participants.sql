-- migrate:up
ALTER TABLE tournament_participants
ADD COLUMN entry_stake DECIMAL DEFAULT 0;

-- migrate:down
ALTER TABLE tournament_participants
DROP COLUMN entry_stake;