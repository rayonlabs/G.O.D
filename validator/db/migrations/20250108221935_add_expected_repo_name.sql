-- migrate:up

ALTER TABLE task_nodes ADD COLUMN expected_repo_name VARCHAR(255);
-- migrate:down

ALTER TABLE task_nodes DROP COLUMN expected_repo_name;
