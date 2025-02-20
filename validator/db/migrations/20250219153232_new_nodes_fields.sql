-- migrate:up

-- Add new columns with default 0
ALTER TABLE nodes
    ADD COLUMN alpha_stake FLOAT NOT NULL DEFAULT 0,
    ADD COLUMN tao_stake FLOAT NOT NULL DEFAULT 0;

-- Remove the defaults while keeping NOT NULL constraint
ALTER TABLE nodes
    ALTER COLUMN alpha_stake DROP DEFAULT,
    ALTER COLUMN tao_stake DROP DEFAULT;

-- migrate:down

ALTER TABLE nodes
    DROP COLUMN alpha_stake,
    DROP COLUMN tao_stake;
