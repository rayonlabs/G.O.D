-- migrate:up
ALTER TABLE tasks RENAME COLUMN hf_training_repo TO training_data;

-- migrate:down
ALTER TABLE tasks RENAME COLUMN training_data TO hf_training_repo;

