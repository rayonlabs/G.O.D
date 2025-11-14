-- migrate:up

ALTER TABLE tournaments
ADD COLUMN previous_final_emission DOUBLE PRECISION;

-- migrate:down

ALTER TABLE tournaments
DROP COLUMN IF EXISTS previous_final_emission;
