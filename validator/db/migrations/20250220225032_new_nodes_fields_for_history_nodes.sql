-- migrate:up
ALTER TABLE nodes_history
    ADD COLUMN alpha_stake FLOAT NOT NULL DEFAULT 0,
    ADD COLUMN tao_stake FLOAT NOT NULL DEFAULT 0;

ALTER TABLE nodes_history
    ALTER COLUMN alpha_stake DROP DEFAULT,
    ALTER COLUMN tao_stake DROP DEFAULT;

-- migrate:down

ALTER TABLE nodes_history
    DROP COLUMN alpha_stake,
    DROP COLUMN tao_stake;
