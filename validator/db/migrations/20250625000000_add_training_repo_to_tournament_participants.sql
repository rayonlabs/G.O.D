-- migrate:up

ALTER TABLE tournament_participants 
ADD COLUMN IF NOT EXISTS training_repo TEXT,
ADD COLUMN IF NOT EXISTS training_commit_hash TEXT;

-- migrate:down

ALTER TABLE tournament_participants 
DROP COLUMN IF EXISTS training_repo,
DROP COLUMN IF EXISTS training_commit_hash; 