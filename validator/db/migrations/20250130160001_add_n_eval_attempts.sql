-- migrate:up

ALTER TABLE tasks ADD COLUMN n_eval_attempts INT NOT NULL DEFAULT 0;

-- migrate:down

ALTER TABLE tasks DROP COLUMN n_eval_attempts;
