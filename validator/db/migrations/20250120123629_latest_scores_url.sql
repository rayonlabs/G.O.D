-- migrate:up

CREATE TABLE latest_scores_url (
    id SERIAL PRIMARY KEY,
    url TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    expired_at TIMESTAMP DEFAULT NULL
);

-- migrate:down

DROP TABLE latest_scores_url;
