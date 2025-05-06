-- migrate:up
ALTER TABLE nodes ADD COLUMN is_blacklisted BOOLEAN DEFAULT FALSE;

-- migrate:down
ALTER TABLE nodes DROP COLUMN is_blacklisted;
