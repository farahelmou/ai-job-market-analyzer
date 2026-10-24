
-- =========================================================
-- AI Job Market Analyzer
-- Phase 7: SQL Analysis with PostgreSQL
-- Dataset: Cleaned AI job postings
-- =========================================================

-- 1. Create the jobs table
CREATE TABLE IF NOT EXISTS jobs (
    job_title TEXT,
    company TEXT,
    region TEXT,
    skills_required TEXT,
    is_remote TEXT,
    is_fresher TEXT,
    job_category TEXT
);

-- 2. Total number of job postings
SELECT COUNT(*) AS total_jobs
FROM jobs;

-- 3. Job postings by category
SELECT
    job_category,
    COUNT(*) AS total_jobs
FROM jobs
GROUP BY job_category
ORDER BY total_jobs DESC;

-- 4. Top 10 regions
SELECT
    region,
    COUNT(*) AS total_jobs
FROM jobs
GROUP BY region
ORDER BY total_jobs DESC
LIMIT 10;

-- 5. Top 10 companies
SELECT
    company,
    COUNT(*) AS total_jobs
FROM jobs
GROUP BY company
ORDER BY total_jobs DESC
LIMIT 10;

-- 6. Remote versus non-remote postings
SELECT
    is_remote,
    COUNT(*) AS total_jobs
FROM jobs
GROUP BY is_remote
ORDER BY total_jobs DESC;

-- 7. Fresher status distribution
SELECT
    is_fresher,
    COUNT(*) AS total_jobs
FROM jobs
GROUP BY is_fresher
ORDER BY total_jobs DESC;

-- 8. Top 15 skills across postings
SELECT
    LOWER(TRIM(skill)) AS skill,
    COUNT(*) AS total_mentions
FROM jobs,
     unnest(string_to_array(skills_required, ',')) AS skill
WHERE LOWER(TRIM(skills_required)) <> 'unknown'
  AND TRIM(skill) <> ''
GROUP BY LOWER(TRIM(skill))
ORDER BY total_mentions DESC
LIMIT 15;

-- 9. Skill mentions by job category
SELECT
    job_category,
    LOWER(TRIM(skill)) AS skill,
    COUNT(*) AS total_mentions
FROM jobs,
     unnest(string_to_array(skills_required, ',')) AS skill
WHERE LOWER(TRIM(skills_required)) <> 'unknown'
  AND TRIM(skill) <> ''
GROUP BY job_category, LOWER(TRIM(skill))
ORDER BY job_category, total_mentions DESC;

-- 10. Top five skills per job category
WITH skill_counts AS (
    SELECT
        job_category,
        LOWER(TRIM(skill)) AS skill,
        COUNT(*) AS total_mentions
    FROM jobs,
         unnest(string_to_array(skills_required, ',')) AS skill
    WHERE LOWER(TRIM(skills_required)) <> 'unknown'
      AND TRIM(skill) <> ''
    GROUP BY job_category, LOWER(TRIM(skill))
),
ranked_skills AS (
    SELECT
        job_category,
        skill,
        total_mentions,
        ROW_NUMBER() OVER (
            PARTITION BY job_category
            ORDER BY total_mentions DESC, skill
        ) AS skill_rank
    FROM skill_counts
)
SELECT
    job_category,
    skill_rank,
    skill,
    total_mentions
FROM ranked_skills
WHERE skill_rank <= 5
ORDER BY job_category, skill_rank;
