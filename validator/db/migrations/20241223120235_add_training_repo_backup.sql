-- migrate:up
ALTER TABLE tasks ADD COLUMN training_repo_backup TEXT DEFAULT NULL;

-- migrate:down
ALTER TABLE tasks DROP COLUMN training_repo_backup;
