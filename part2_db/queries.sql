-- Query 1: Conversion rate by lead source, sources with 200+ leads, best first.

SELECT
    source,
    COUNT(*)                                   AS total_leads,
    SUM(CASE WHEN converted THEN 1 ELSE 0 END) AS converted_leads,
    ROUND(100.0 * SUM(CASE WHEN converted THEN 1 ELSE 0 END) / COUNT(*), 2) AS conversion_rate_pct
FROM leads
GROUP BY source
HAVING COUNT(*) >= 200
ORDER BY conversion_rate_pct DESC;


-- Query 2: Find duplicate leads (same lead entered twice by different agents).
--
-- The tell in this data: crm_record_hash is a fingerprint of the underlying
-- person, but lead_id is a fresh ID per row (e.g. 'MGC-104974' and
-- 'MGC-104974-B' — same person, two agents, two rows, identical hash,
-- identical created_at). So "duplicate" = same crm_record_hash, different
-- lead_id.

SELECT
    crm_record_hash,
    COUNT(*)                       AS entry_count,
    STRING_AGG(lead_id, ', ')      AS lead_ids   -- GROUP_CONCAT(lead_id, ', ') in MySQL/SQLite
FROM leads
GROUP BY crm_record_hash
HAVING COUNT(*) > 1
ORDER BY entry_count DESC;

-- Prevention at the schema level: crm_record_hash has a UNIQUE constraint in
-- schema.sql. The correct long-term fix is upstream of the DB though — the
-- CRM/intake form should compute the hash (normalized phone + CNIC, not name,
-- since names collide and aren't unique) BEFORE insert and do a lookup-or-
-- create, so a second agent entering the same buyer gets routed to the
-- existing lead_id instead of the insert failing after the fact.
