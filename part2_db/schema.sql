-- MGC leads schema
-- One table, defended below. The CSV is one flat event per lead with no
-- repeating groups (no multiple calls-per-row, no multiple visits-per-row —
-- those are pre-aggregated counts), so normalizing further just adds joins
-- with no gain. Two small lookup tables ARE worth splitting out because
-- they carry a controlled vocabulary that should be validated at insert time,
-- not left as free-text strings.

CREATE TABLE lead_sources (
    source_name TEXT PRIMARY KEY
);
INSERT INTO lead_sources (source_name) VALUES
    ('Facebook Ads'), ('Property Portal'), ('Google Search'), ('Instagram'),
    ('Referral'), ('Walk-in'), ('WhatsApp Campaign'), ('Expo Stall'), ('Billboard');

CREATE TABLE property_types (
    type_name TEXT PRIMARY KEY
);
INSERT INTO property_types (type_name) VALUES
    ('Apartment'), ('Plot'), ('Villa'), ('Commercial Shop'), ('Penthouse'), ('Farmhouse');

CREATE TABLE leads (
    lead_id                        TEXT PRIMARY KEY,
    -- crm_record_hash is a fingerprint of the underlying person/enquiry
    -- (e.g. hash of normalized phone + name). UNIQUE here is the schema-level
    -- fix for the duplicate problem in queries.sql: two agents entering the
    -- same person today produces two lead_ids but the SAME hash, and this
    -- constraint rejects the second insert outright instead of letting it
    -- through and relying on a nightly dedup job to catch it later.
    crm_record_hash                BIGINT NOT NULL UNIQUE,

    created_at                     TIMESTAMP NOT NULL,
    source                         TEXT NOT NULL REFERENCES lead_sources(source_name),
    city                           TEXT NOT NULL,   -- see queries.sql note: needs normalization (casing/abbrev.), not enforced here
    area                           TEXT,             -- nullable: not always captured at intake
    property_type                  TEXT NOT NULL REFERENCES property_types(type_name),
    budget_pkr_lac                 NUMERIC(10,2),    -- nullable: buyer doesn't always disclose upfront
    bedrooms                       SMALLINT,         -- nullable: not applicable to Plot / Commercial Shop

    first_response_minutes         INTEGER,          -- nullable: not yet contacted
    calls_made                     INTEGER NOT NULL DEFAULT 0,
    total_call_seconds             INTEGER NOT NULL DEFAULT 0,
    whatsapp_replies               INTEGER NOT NULL DEFAULT 0,
    site_visits                    INTEGER NOT NULL DEFAULT 0,

    agent_experience_years         NUMERIC(4,1),     -- nullable: unassigned lead
    is_overseas                    BOOLEAN NOT NULL DEFAULT FALSE,
    referred_by_existing_client    BOOLEAN NOT NULL DEFAULT FALSE,
    has_financing_approved         BOOLEAN NOT NULL DEFAULT FALSE,

    -- Set only once a token is actually received — i.e. only meaningful
    -- post-conversion. Kept here as an operational record, but flagged in
    -- Part 3 as NOT a usable model feature (it's a symptom of conversion,
    -- not a predictor of it).
    token_amount_received_pkr      NUMERIC(12,2) NOT NULL DEFAULT 0,

    converted                      BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX idx_leads_source ON leads(source);
CREATE INDEX idx_leads_created_at ON leads(created_at);
CREATE INDEX idx_leads_converted ON leads(converted);
