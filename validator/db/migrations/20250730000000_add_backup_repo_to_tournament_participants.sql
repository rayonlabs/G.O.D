-- migrate:up
ALTER TABLE tournament_participants
ADD COLUMN backup_repo TEXT;

-- migrate:down
ALTER TABLE tournament_participants
DROP COLUMN backup_repo; 