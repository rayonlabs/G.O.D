-- migrate:up
ALTER TABLE tournament_participants
ADD COLUMN stake_required DECIMAL DEFAULT 0;

-- migrate:down
ALTER TABLE tournament_participants
DROP COLUMN stake_required;