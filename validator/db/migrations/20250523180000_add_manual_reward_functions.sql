-- migrate:up
ALTER TABLE reward_functions
ADD COLUMN is_manual BOOLEAN NOT NULL DEFAULT FALSE;

-- migrate:down
ALTER TABLE reward_functions
DROP COLUMN is_manual;
