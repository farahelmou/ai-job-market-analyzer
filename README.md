# AI Job Market Analyzer

**An end-to-end data analysis and unsupervised machine learning project exploring AI/ML job postings, skill requirements, and job-market patterns in India.**

## Overview

The **AI Job Market Analyzer** explores job postings across three categories—**Data Analyst, Data Scientist, and ML Engineer**—to understand the information available in job listings and identify patterns in their recorded skill requirements.

The project follows a practical data workflow: combining source datasets, inspecting and cleaning records, performing exploratory data analysis (EDA), transforming skill lists into machine-learning features, applying K-Means clustering, and using PostgreSQL to query and summarize the cleaned data.

The project is intended as a learning and portfolio project. Its findings describe the collected dataset; they are not a real-time vacancy tracker or a complete representation of the job market.

## Objectives

- Understand the structure and quality of job-posting data.
- Combine listings from three job categories into one dataset.
- Identify missing values, inconsistent entries, and exact duplicates.
- Explore job categories, locations, companies, remote-work labels, fresher status, and recorded skills.
- Convert text-based skill lists into binary features and text-vector representations.
- Explore groups of postings with similar recorded skills using K-Means.
- Evaluate clustering with the elbow method and silhouette score.
- Use PostgreSQL and SQL to produce reproducible summaries and skill rankings.
- Document the workflow, results, limitations, and possible extensions.

## Dataset

The source data consists of three CSV files in `Data/raw/`.

| Source file | Initial rows | Job category |
|---|---:|---|
| `Data_Analyst_jobs.csv` | 665 | Data Analyst |
| `Data_Science_jobs.csv` | 676 | Data Scientist |
| `ML_Engineer_jobs.csv` | 162 | ML Engineer |
| **Total** | **1,503** | |

The combined dataset includes job title, company, region, required skills, remote-work status, and fresher status. A `job_category` field is added during combination to identify the source category.

### Main fields

| Field | Description |
|---|---|
| `job title` | Title of the advertised position |
| `company` | Company associated with the posting |
| `region` | Region or location label in the source data |
| `skills required` | Comma-separated skill labels recorded for the posting |
| `is remote?` | Remote-work indicator |
| `is fresher?` | Fresher-status indicator |
| `job_category` | Data Analyst, Data Scientist, or ML Engineer |

### Geographic scope

The listings represent job postings from **India**. Region values include city names as well as labels such as `Remote` and `India`; therefore, not every region value corresponds to a city.

## Tools and Technologies

- **Python** — general-purpose analysis and data processing
- **Pandas and NumPy** — data manipulation and numerical operations
- **Matplotlib** — visualizations
- **Scikit-learn** — text vectorization, K-Means, and clustering evaluation
- **Jupyter Notebook** — interactive analysis and experimentation
- **PostgreSQL and SQL** — structured querying and aggregation
- **Git and GitHub** — version control and project sharing

## Repository Structure

```text
ai-job-market-analyzer/
├── Data/
│   ├── raw/
│   │   ├── Data_Analyst_jobs.csv
│   │   ├── Data_Science_jobs.csv
│   │   └── ML_Engineer_jobs.csv
│   └── processed/
│       ├── Ai_jobs.csv
│       ├── Ai_jobs_cleaned.csv
│       ├── ai_jobs_clean.csv
│       ├── ai_jobs_skill_features.csv
│       ├── ai_jobs_clustered.csv
│       └── cluster_summary.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_job_clustering.ipynb
│   └── elbow.png
├── sql/
│   └── queries.sql
├── .gitignore
└── README.md
```

The processed folder contains intermediate and final artifacts from different stages of the workflow. The notebooks are the primary record of the analysis and code.

## Workflow

### Phase 1 — Data Understanding and Exploration

**Notebook:** [`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb)

The initial phase examines the three source files and combines the records into one dataset.

Activities include:
- Loading and inspecting the source CSV files.
- Checking dimensions, columns, data types, and sample records.
- Comparing the source datasets and understanding their fields.
- Adding a job-category label to preserve the type of posting.
- Combining the listings into a single dataset for later analysis.

**Output:** `Data/processed/Ai_jobs.csv`  
**Initial combined size:** 1,503 rows.

### Phase 2 — Data Cleaning and Preparation

**Notebook:** [`notebooks/02_data_cleaning.ipynb`](notebooks/02_data_cleaning.ipynb)

This phase prepares the combined dataset for analysis.

Activities include:
- Inspecting the dataset structure and values.
- Reviewing missing and unknown information.
- Identifying and removing exact duplicate records.
- Preparing a cleaned version of the dataset for subsequent EDA and modeling.

The intermediate file `Ai_jobs_cleaned.csv` contains 1,503 rows. After the cleaning and duplicate-removal steps in the later EDA workflow, `ai_jobs_clean.csv` contains **1,361 rows**.

**Final cleaned dataset:** `Data/processed/ai_jobs_clean.csv`

#### Missing information

- **Required skills:** 788 of 1,361 cleaned postings have unknown skill information (**57.9%**).
- **Fresher status:** 927 of 1,361 postings have unknown fresher status (**68.1%** in the imported SQL dataset).

Unknown information is not the same as a confirmed negative value. In particular, a posting with unknown skills should not be interpreted as a posting that requires no skills.

### Phase 3 — Exploratory Data Analysis (EDA)

**Notebook:** [`notebooks/03_eda.ipynb`](notebooks/03_eda.ipynb)

EDA is used to examine the cleaned data and understand the distributions and data-quality issues before modeling.

The analysis explores:
- Job-posting counts by category.
- Geographic distribution and frequently represented regions.
- Companies with the largest numbers of listings in the dataset.
- Remote versus non-remote labels.
- Known and unknown fresher status.
- Skill information and its availability.

EDA helps identify the limits of the dataset—for example, the large share of unknown skill entries—and informs the decision to build skill-based features only for postings with recorded skills.

### Phase 4 — Skill Preprocessing and Feature Engineering

**Notebook:** [`notebooks/04_job_clustering.ipynb`](notebooks/04_job_clustering.ipynb)

The skill preparation and clustering work are currently developed together in the fourth notebook.

#### 4.1 Normalize skill text

A working field named `skills_text` is prepared from `skills required`. The preprocessing standardizes capitalization and whitespace, normalizes unknown entries, and splits comma-separated values into individual skills.

#### 4.2 Build the skill vocabulary

The notebook identifies **26 distinct skill labels** in the recorded skill lists. The vocabulary includes labels such as Python, SQL, Machine Learning, Deep Learning, AI, Statistics (`Stat`), Big Data, Data Mining, Database, Data Visualization, Excel, ETL, Deployment, PyTorch, TensorFlow, NLP, and Computer Vision.

These are the skill labels present in this dataset, not an exhaustive list of industry skills.

#### 4.3 Create binary skill features

Each posting with recorded skills is represented by a binary vector:
- `1` means the skill appears in the posting's recorded skill list.
- `0` means the skill does not appear in that recorded list.

Postings with unknown skills are excluded from the skill-only modeling matrix rather than being treated as if they have no skills.

| Feature-engineering result | Value |
|---|---:|
| Cleaned postings | 1,361 |
| Postings with recorded skills | 573 |
| Postings with unknown skills | 788 |
| Binary skill features | 26 |
| Binary matrix used for clustering | 573 × 26 |

The notebook also experiments with text representations:

- **Bag of Words (BoW):** represents the frequency of tokens in each recorded skill string.
- **TF-IDF:** weights tokens according to their distribution across the skill strings.

The BoW and TF-IDF matrices each contain **573 rows and 31 tokens**. They are explored as alternative representations; the documented K-Means experiment uses the binary skill feature matrix.

**Output:** `Data/processed/ai_jobs_skill_features.csv`

### Phase 5 — K-Means Clustering

**Notebook:** [`notebooks/04_job_clustering.ipynb`](notebooks/04_job_clustering.ipynb)

K-Means is applied to the binary skill matrix to explore whether postings can be grouped according to similar recorded skill profiles.

#### Selecting the number of clusters

The notebook tests values of \(k\) from 2 to 10 and examines:
- **The elbow method**, using inertia to inspect how within-cluster variation changes with \(k\).
- **The silhouette score**, which compares how similar an observation is to its assigned cluster versus other clusters.

The silhouette scores increase across the tested values:

| Number of clusters (\(k\)) | Silhouette score |
|---:|---:|
| 2 | 0.1669 |
| 3 | 0.1950 |
| 4 | 0.2139 |
| 5 | 0.2536 |
| 6 | 0.2875 |
| 7 | 0.3360 |
| 8 | 0.3603 |
| 9 | 0.4023 |
| 10 | 0.4237 |

For the documented experiment, the selected configuration is **K-Means with 10 clusters**, using `random_state=42` and `n_init=10`.

| Model detail | Value |
|---|---|
| Algorithm | K-Means |
| Input data | Binary skill features |
| Postings modeled | 573 |
| Features | 26 |
| Clusters | 10 |
| Silhouette score | 0.4237 |

A higher silhouette score among the tested values does not, by itself, prove that 10 is the universally optimal number of clusters. Cluster usefulness and stability would need further evaluation.

#### Cluster sizes

| Cluster | Number of postings |
|---:|---:|
| 0 | 46 |
| 1 | 57 |
| 2 | 64 |
| 3 | 29 |
| 4 | 46 |
| 5 | 58 |
| 6 | 51 |
| 7 | 55 |
| 8 | 33 |
| 9 | 134 |
| **Total** | **573** |

#### Interpreting cluster profiles

The clusters are examined by calculating the percentage of postings in each cluster that contain each skill. The dominant recorded patterns include:

- Cluster 0 — Database
- Cluster 1 — Statistics
- Cluster 2 — Python
- Cluster 3 — Data Structures
- Cluster 4 — AI
- Cluster 5 — Machine Learning
- Cluster 6 — Data Mining
- Cluster 7 — Data Visualization
- Cluster 8 — Excel
- Cluster 9 — a more varied set of skills, with no single skill reaching 20% prevalence

These are **skill-centered groups**, not definitive job-role classifications. A cluster dominated by Python does not necessarily correspond to one particular occupation; it describes similarity under the selected binary features and K-Means configuration.

**Outputs:**
- `Data/processed/ai_jobs_clustered.csv`
- `Data/processed/cluster_summary.csv`
- `notebooks/elbow.png`

### Phase 6 — SQL Analysis with PostgreSQL

**SQL file:** [`sql/queries.sql`](sql/queries.sql)

The cleaned dataset is imported into PostgreSQL for structured analysis. The database is named `ai_job_market` and the table is named `jobs`.

The `jobs` table contains:
- `job_title`
- `company`
- `region`
- `skills_required`
- `is_remote`
- `is_fresher`
- `job_category`

The SQL script includes queries for:
1. Total number of postings.
2. Postings by job category.
3. Top regions.
4. Top companies.
5. Remote-work distribution.
6. Fresher-status distribution.
7. Most frequently recorded skills.
8. Skill mentions by job category.
9. The top five recorded skills in each category.

SQL techniques include `COUNT()`, `GROUP BY`, `ORDER BY`, `LIMIT`, `LOWER()`, `TRIM()`, `string_to_array()`, `unnest()`, common table expressions (CTEs), and `ROW_NUMBER()` with `PARTITION BY`.

#### SQL findings

**Job categories**

| Category | Postings |
|---|---:|
| Data Scientist | 620 |
| Data Analyst | 585 |
| ML Engineer | 156 |
| **Total** | **1,361** |

**Most represented regions**

| Region | Postings |
|---|---:|
| Bengaluru | 375 |
| Hyderabad | 114 |
| Mumbai | 99 |
| Pune | 91 |
| Gurgaon | 85 |
| Remote | 64 |
| India | 64 |
| Chennai | 55 |
| Noida | 39 |
| Kolkata | 28 |

**Companies with the most postings**

| Company | Postings |
|---|---:|
| Accenture | 25 |
| EY | 15 |
| JPMorgan Chase & Co. | 15 |
| KPMG | 10 |
| Bosch Group | 9 |

Mercedes-Benz Research and Development India and Deloitte each have 8 postings in the dataset.

**Remote-work labels**

| Label | Postings | Share |
|---|---:|---:|
| Non-remote | 1,179 | 86.6% |
| Remote | 182 | 13.4% |
| **Total** | **1,361** | **100%** |

**Fresher-status labels**

| Label | Postings |
|---|---:|
| Unknown | 927 |
| No | 431 |
| Yes | 3 |
| **Total** | **1,361** |

**Most frequently recorded skills**

| Skill label | Mentions |
|---|---:|
| Statistics (`stat`) | 100 |
| Python | 88 |
| Machine Learning | 62 |
| Data Mining | 58 |
| Data Visualization | 57 |
| Database | 52 |
| AI | 47 |
| Big Data | 43 |
| Excel | 41 |
| Deep Learning | 34 |
| SQL | 33 |
| Data Structures | 29 |
| ETL | 25 |
| Deployment | 20 |
| Dashboard | 17 |

**Top recorded skills by category**

| Category | Top recorded skills |
|---|---|
| Data Analyst | Database, Data Visualization, Excel, Statistics, Python |
| Data Scientist | Python, Machine Learning, Statistics, Data Mining, Big Data |
| ML Engineer | AI, Deep Learning, Deployment, PyTorch, Statistics |

Skill counts refer to mentions in the recorded skill lists. Because many postings have unknown skill information, these counts should not be treated as a complete measurement of skill demand across all vacancies.

## Overall Findings

The analysis highlights several patterns in the collected data:

- Data Scientist and Data Analyst postings make up most of the cleaned dataset.
- Bengaluru is the most frequently represented region label.
- Most postings are marked as non-remote.
- Unknown values are common, particularly in the required-skills and fresher-status fields.
- Python, Statistics, and Machine Learning are among the most frequently recorded skill labels.
- Skill profiles differ across the three job categories in the dataset.
- K-Means identified ten groups of postings with similar recorded skill patterns, with a silhouette score of 0.4237 for the selected experiment.
- PostgreSQL queries provide a reproducible way to summarize the dataset and rank skill mentions.

## Limitations

- **Geographic coverage:** The dataset represents India and cannot be generalized directly to the global job market.
- **Missing skill data:** 57.9% of cleaned postings have unknown skill information. Skill analyses and clustering therefore cover only a subset of all postings.
- **Missing fresher data:** 68.1% of imported postings have unknown fresher status, so the dataset cannot support strong conclusions about entry-level opportunities.
- **Category imbalance:** The number of postings differs substantially by category.
- **Skill normalization:** Labels such as `Stat` and `Python` are recorded dataset terms; synonyms and differently worded skills may not be unified.
- **Clustering assumptions:** K-Means groups postings using binary skill features. The resulting clusters are not ground-truth roles, and a single evaluation score cannot establish model quality on its own.
- **Dataset coverage and freshness:** Counts describe the available records, not every vacancy published by a company or the current job market.
- **Region labels:** Values such as `Remote` and `India` are included alongside city names and should not be interpreted as cities.

## Future Improvements

Possible extensions include:

- Gathering a larger, more recent dataset from additional sources and regions.
- Improving skill normalization with a curated taxonomy and synonym mapping.
- Exploring NLP-based skill extraction from full job descriptions, if those descriptions become available.
- Comparing K-Means with alternative clustering methods and additional evaluation criteria.
- Studying cluster stability and the effect of different feature representations.
- Visualizing clusters with dimensionality-reduction methods.
- Building an interactive dashboard for exploring roles, locations, and skill requirements.
- Developing a job-to-skill matching or recommendation feature after validating the data and evaluation approach.
- Automating data collection and preprocessing so that the analysis can be updated regularly.

## Getting Started

### Prerequisites

- Python 3
- Jupyter Notebook
- PostgreSQL for the SQL phase

### Install Python dependencies

Create and activate a virtual environment, then install the packages used by the notebooks:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib scikit-learn jupyter
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

### Run the notebooks

From the repository root:

```bash
jupyter notebook
```

Run the notebooks in order:

1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_data_cleaning.ipynb`
3. `notebooks/03_eda.ipynb`
4. `notebooks/04_job_clustering.ipynb`

The original CSV files are stored in `Data/raw/`. The processed files are stored in `Data/processed/`.

**Reproducibility note:** the current notebooks use both `Data/` and `data/` in some file paths, and the data-exploration notebook uses a slightly different capitalization for the Data Science CSV filename. These path spellings should be made consistent with the actual repository names before rerunning the workflow on a case-sensitive filesystem.

### Run the SQL analysis

Start PostgreSQL and create the database if it does not already exist:

```sql
CREATE DATABASE ai_job_market;
```

Connect to the database:

```bash
psql -d ai_job_market
```

Create the `jobs` table using the schema at the top of `sql/queries.sql`. From the repository root, import the cleaned dataset (adjust the CSV path only if your local directory differs):

```sql
\copy jobs (job_title, company, region, skills_required, is_remote, is_fresher, job_category) FROM 'Data/processed/ai_jobs_clean.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');
```

Then run the SQL script from the repository root:

```bash
psql -d ai_job_market -f sql/queries.sql
```

**Important:** import the CSV only when you need to populate the table. Reimporting it into an already populated table without clearing or replacing the existing rows can duplicate records. If the database is already populated, run the queries without importing the CSV again.

## Key Learnings

This project provided hands-on experience with:

- Inspecting and combining real-world CSV datasets.
- Cleaning duplicate records and handling missing or unknown values.
- Conducting exploratory data analysis.
- Normalizing text-based skill lists.
- Engineering binary features and exploring BoW and TF-IDF representations.
- Applying K-Means and interpreting unsupervised clustering results.
- Evaluating clusters with inertia and silhouette scores.
- Creating a PostgreSQL table and importing processed data.
- Writing SQL queries for aggregation, ranking, and category-level comparisons.
- Communicating results while documenting data limitations.

## Conclusion

The AI Job Market Analyzer brings together data preparation, exploratory analysis, feature engineering, unsupervised learning, and SQL into one end-to-end project. It demonstrates how raw job postings can be transformed into structured summaries and skill-based groupings.

The findings provide an initial view of the recorded job postings and highlight the importance of data quality when interpreting job-market patterns. The project also establishes a foundation for future work in skill normalization, richer text analysis, clustering evaluation, and interactive job-market exploration.

---

**Author:** Farah Elmourabite  
**GitHub:** [farahelmou](https://github.com/farahelmou)
