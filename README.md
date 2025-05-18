# PANORAMA Data Generation Pipeline

**PANORAMA** (Profile-based Assemblage for Naturalistic Online Representation and Attribute Memorization Analysis) is a synthetic data pipeline developed for generating realistic, PII-rich datasets that simulate online human behavior. This repository contains all components required to create, process, and manage synthetic data suitable for pretraining and privacy-risk analysis in LLMs.

[PANORAMA as HuggingFace dataset](https://huggingface.co/datasets/srirxml/PANORAMA)

## 📁 Project Structure

```
DataGeneration/
├── Data/                            # Stores all generated intermediate and final files
├── Project/
│   ├── WikiSeedDataGeneration.py
│   ├── SyntheticProfileGenerator.py
│   ├── PromptSyntheticDataGenTemplatedApproachV1.md
│   ├── SubmitToAzureFoundry.py
│   ├── ExtractGeneratedSyntheticPassage.py
│   ├── SyntheticSocialDataGenPrompt.md
│   ├── AzureGenerateSyntheticTrainingData.py
│   └── process_synthetic_data.py
```

---

## 🔄 Pipeline Overview

### 1. Wikipedia-Inspired Article Seeding

**Script:** `WikiSeedDataGeneration.py`  
**Input:** [HuggingFace dataset](https://huggingface.co/datasets/wikimedia/structured-wikipedia)  
**Extracted Fields:**
- Name
- URL
- Abstract
- Wikipedia Content
- Personal Life
- Early Life

**Output:**
- `wikipedia_people_{timestamp}.tsv`
- `wikipedia_people_{timestamp}.jsonl`

---

### 2. Synthetic Profile Generation

**Script:** `SyntheticProfileGenerator.py`  
**Function:** Generates synthetic human profiles using [Faker](https://faker.readthedocs.io/en/master/) for 8 English locales:  
`['en_US', 'en_GB', 'en_CA', 'en_AU', 'en_NZ', 'en_IE', 'en_IN', 'en_PH']`

**Output:**  
- `synthetic_profiles_{timestamp}.tsv`

---

### 3. Article Synthesis (Step 1)

**Template:** `PromptSyntheticDataGenTemplatedApproachV1.md`  
**Input:**  
- Synthetic Profile JSON (from Step 2)  
- Wikipedia Inspiration Text (from Step 1)

**Function:** Combines structured profiles with Wikipedia-style narrative cues to synthesize realistic biographies.

---

### 4. Article Generation via Azure OpenAI

**Script:** `SubmitToAzureFoundry.py`  
**Model Used:** Azure AI Foundry (OpenAI o3-mini)  
**Input:** Prompt from Step 3  
**Output:**
- Raw: `azure_batch_job_{timestamp}.live.jsonl`
- Aggregated: `generated_passages_azure_results.10K.jsonl`

---

### 5. Extracting Synthetic Articles

**Script:** `ExtractGeneratedSyntheticPassage.py`  
**Function:** Isolates the `[Synthetic Article]` section from the generated outputs for downstream use.  
**Output:**  
- `extracted_passages_azure_results.10K.jsonl`

---

### 6. Generating Synthetic Social Content

**Template:** `SyntheticSocialDataGenPrompt.md`  
**Script:** `AzureGenerateSyntheticTrainingData.py`  
**Input:**  
- `extracted_passages_azure_results.10K.jsonl`  
- Synthetic profiles

**Generated Formats:**  
- Social Media Posts  
- Forum Posts  
- Online Reviews  
- Blog/News Comments  
- Online Marketplace Listings  

**Output:**
- `Azure_Synthetic_Data_10K.{split}_{timestamp}.live_output.jsonl`  
- Combined: `Azure_Synthetic_Data_10K.combined.jsonl`

---

### 7. Data Post-Processing

**Script:** `process_synthetic_data.py`  
**Function:** Cleans, normalizes, and prepares the dataset for pretraining.  
**Final Output:**
- Processed: `Azure_Synthetic_Data_10K.processed.jsonl`  
- HF Pretraining Format: `Azure_Synthetic_Data_10K.pretraining.hf.tsv`

---

## 📊 Dataset Summary

- **Profiles Generated:** ~9,674  
- **Synthetic Articles:** 9,674  
- **Total Synthetic Samples:** ~384,789  
- **Content Types:**  
  - Wikipedia-style Biographies  
  - Social Media Posts  
  - Forum Threads  
  - Online Reviews  
  - Blog Comments  
  - Marketplace Listings  

---

## 📦 Dependencies

- Python 3.8+
- `Faker`
- `datasets`
- OpenAI SDK (for inference)
---

## 🔐 Ethics & Usage

PANORAMA is a **synthetic dataset** developed with no use of real PII. It is intended **solely for research** into privacy risks, model memorization, and responsible AI development. Do not use this dataset for real-world PII applications.

---