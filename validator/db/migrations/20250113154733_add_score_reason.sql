-- migrate:up
ALTER TABLE task_nodes
ADD COLUMN score_reason TEXT;

-- migrate:down
ALTER TABLE task_nodes
DROP COLUMN score_reason;
