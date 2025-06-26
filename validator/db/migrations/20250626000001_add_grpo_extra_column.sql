-- migrate:up

ALTER TABLE grpo_tasks ADD COLUMN extra_column TEXT;

-- migrate:down

ALTER TABLE grpo_tasks DROP COLUMN extra_column;