# PANORAMA: Profile-based Assemblage for Naturalistic Online Representation and Attribute Memorization Analysis

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2505.12238-b31b1b.svg)](https://arxiv.org/abs/2505.12238)
[![Dataset](https://img.shields.io/badge/Dataset-PANORAMA-blue.svg)](https://huggingface.co/datasets/srirxml/PANORAMA)
[![Dataset Plus](https://img.shields.io/badge/Dataset-PANORAMA--Plus-green.svg)](https://huggingface.co/datasets/srirxml/PANORAMA-Plus)
[![Website](https://img.shields.io/badge/Website-PANORAMA%20Privacy%20Dataset-orange.svg)](https://panorama-privacy-dataset.github.io/)

**PANORAMA** is a comprehensive synthetic dataset designed to provide realistic, PII-rich datasets that simulate online human behavior for studying sensitive data memorization in Large Language Models (LLMs). This repository contains the complete pipeline for generating, processing the synthetic data suitable for privacy-risk analysis and responsible AI development.

## 📚 Resources

- **Paper**: [PANORAMA: A synthetic PII-laced dataset for studying sensitive data memorization in LLMs](https://arxiv.org/abs/2505.12238)
- **Main Dataset**: [PANORAMA on Hugging Face](https://huggingface.co/datasets/srirxml/PANORAMA) (384,789 samples)
- **Profile Dataset**: [PANORAMA-Plus on Hugging Face](https://huggingface.co/datasets/srirxml/PANORAMA-Plus) (9,674 profiles)
- **Project Website**: [PANORAMA Privacy Dataset](https://panorama-privacy-dataset.github.io/)

## 🎯 Overview

PANORAMA generates fully synthetic datasets that closely emulate the distribution, variety, and context of personally identifiable information (PII) as it naturally occurs in online environments. The pipeline creates internally consistent synthetic profiles and generates diverse content types including:

- **Wikipedia-style biographies** with realistic life narratives by feeding profile and a real wikipedia article as seed.
- **Social media posts** (Twitter, Facebook, Instagram, etc.)
- **Forum discussions** and technical posts
- **Online reviews** (restaurants, products, services)
- **Blog and news article comments**
- **Online marketplace listings** and classified ads

## 📊 Dataset Statistics

- **Total Synthetic Profiles**: 9,674
- **Total Synthetic Samples**: 384,789
- **Content Types**: 6 distinct categories
- **Geographic Coverage**: 8 English-speaking locales `['en_US', 'en_GB', 'en_CA', 'en_AU', 'en_NZ', 'en_IE', 'en_IN', 'en_PH']`
- **Data Format**: Parquet files optimized for ML workflows

### Content Distribution
- **Social Media Posts**: 88,408 samples (23.0%)
- **Forum Posts**: 83,546 samples (21.7%)
- **Online Reviews**: 79,758 samples (20.7%)
- **Blog/News Comments**: 77,472 samples (20.1%)
- **Online Ads**: 45,936 samples (11.9%)
- **Wikipedia Articles**: 9,674 samples (2.5%)

## 🏗️ Pipeline Architecture

### 1. **Wikipedia Seed Data Generation**
**Script**: `src/generate_wikipedia_seed_data.py`  
**Purpose**: Extracts real Wikipedia articles to provide narrative inspiration and structural patterns for synthetic biographies.
**Wikipedia Article Seed Source**: [Wikimedia Structured Wikipedia Dataset](https://huggingface.co/datasets/wikimedia/structured-wikipedia)  


### 2. **Synthetic Profile Generation**
**Script**: `src/generate_synthetic_profile.py`  
**Purpose**: Creates comprehensive synthetic human profiles using the Faker library across 8 English locales.

**Features**:
- **Personal**: First Name, Last Name, Gender, Age, Nationality, Birth Date, Birth City
- **Relationship**: Father's Name, Mother's Name, Marital Status, Spouse Name, Children Count
- **IDs**: National ID, Passport Number, Driver's License
- **Contact Details**: Phone Number, Work Phone, Address, Email Address, Work Email, Social Media Handles
- **Socio-Economic**: Education Info,  Finance Status, Net Worth, Employer, Job Title, Annual Salary, Credit Score
- **Sensitive Health**: Blood Type, Allergies, Disability, Emergency Contact Name, Emergency Contact Phone


### 3. **Wikipedia-Style Article Synthesis**
**Script**: `src/generate_synthetic_wiki_style_article.py`  
**Prompt**: `prompts/synthetic_article_gen_prompt.md`  
**Model**: Azure OpenAI o3-mini
**Process**: Combines synthetic profiles with Wikipedia inspiration text to generate realistic biographies that maintain factual consistency while incorporating nuanced narrative elements.

#### 3.1 **Extraction of Generated Articles**
**Script**: `src/extract_generated_synthetic_passage.py`  
**Purpose**: Model results contain synthetic passages and seed wiki's usage notes, this module extracts synthetic passage.

#### 3.2 **Clean entity contamination from seed wiki article**
**Script**: `src/clean_generated_synthetic_passage.py`
**Purpose**: Cleans any contaimination such as entity name, dates, location from seed wiki-article in generated synthetic passages.

### 4. **Multi-Format Content Generation**
**Script**: `src/generate_synthetic_online_text_varieties.py`  
**Prompt**: `prompts/synthetic_online_text_gen_prompt.md`  
**Model**: Azure OpenAI o3-mini
**Purpose**: This module generates the various modalities of text that are primary contribution from PANORAMA dataset.

#### 4.1 **Extraction of Generated Content**
**Script**: `src/extract_generated_synthetic_online_data.py`
**Purpose**: Various modalities of text generated are in natural language format, this scripts extracts it into a JSON schema.

#### 4.2 **Fix missing social-id issue**
**Script**: `src/fix_missing_social_id_hf.py`
**Purpose**: Through manual analysis we identified that social media posts generated didn't have any social media handles associated with them in majority of cases. To mitigate this we either add a social handle or first name to the post.


### 5. **Dataset Conversion and Export**
**Script**: `src/convert_tsv_to_hugging_face_dataset.py`  
**Purpose**: Converts processed data to Hugging Face Dataset format for easy integration with ML workflows.

## 📁 Project Structure

```
PANORAMA-DataGen/
├── src/                                    # Core pipeline scripts
│   ├── generate_wikipedia_seed_data.py     # Wikipedia data extraction
│   ├── generate_synthetic_profile.py       # Profile generation
│   ├── generate_synthetic_wiki_style_article.py  # Article synthesis
│   ├── generate_synthetic_online_text_varieties.py  # Multi-format content
│   ├── extract_generated_synthetic_passage.py      # Content extraction
│   ├── extract_generated_synthetic_online_data.py  # Online data processing
│   ├── create_aggregate_profile_content_hf.py     # Data aggregation
│   ├── fix_missing_social_id_hf.py                # Data consistency
│   ├── convert_tsv_to_hugging_face_dataset.py     # HF dataset conversion
│   ├── analyze_and_visualize_dataset.py           # Dataset analysis
│   ├── utilities/                          # Helper utilities
│   │   └── convert_tsv_to_jsonl.py
│   ├── notebooks/                          # Jupyter notebooks
│   │   └── hf_dataset_replicator.ipynb
│   └── misc/                               # Miscellaneous scripts
├── prompts/                                # Generation prompts
│   ├── synthetic_article_gen_prompt.md     # Article generation prompt
│   └── synthetic_online_text_gen_prompt.md # Multi-format content prompt
├── stats/                                  # Dataset statistics
│   └── Stats.md
└── LICENSE.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Azure OpenAI API access
- Required packages: `faker`, `datasets`, `pandas`, `openai`, `tiktoken`

### Installation
```bash
git clone https://github.com/your-username/PANORAMA-DataGen.git
cd PANORAMA-DataGen
pip install -r requirements.txt
```

## 🔬 Research Applications

PANORAMA is designed for systematic study of sensitive data memorization in LLMs and evaluation of privacy-preserving strategies:

- **Memorization Analysis**: Study how LLMs memorize and reproduce PII
- **Privacy Risk Assessment**: Evaluate model privacy vulnerabilities
- **Defense Evaluation**: Test privacy-preserving techniques
- **Benchmark Development**: Create standardized privacy evaluation benchmarks

## 📄 Citation

If you use PANORAMA in your research, please cite:

```bibtex
@article{selvam2025panorama,
  title={PANORAMA: A synthetic PII-laced dataset for studying sensitive data memorization in LLMs},
  author={Selvam, Sriram and Ghosh, Anneswa},
  journal={arXiv preprint arXiv:2505.12238},
  year={2025}
}
```

## 📜 License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). See [LICENSE.md](LICENSE.md) for details.

## ⚠️ Ethical Considerations

**Important**: PANORAMA is a **fully synthetic dataset** developed with no use of real PII. It is intended **solely for research** into privacy risks, model memorization, and responsible AI development. Please do not use for purposes other than research.

## 🔗 Links

- **Paper**: [arXiv:2505.12238](https://arxiv.org/abs/2505.12238)
- **Main Dataset**: [PANORAMA on Hugging Face](https://huggingface.co/datasets/srirxml/PANORAMA)
- **Profile Dataset**: [PANORAMA-Plus on Hugging Face](https://huggingface.co/datasets/srirxml/PANORAMA-Plus)
- **Project Website**: [PANORAMA Privacy Dataset](https://panorama-privacy-dataset.github.io/)

---

**Authors**: Sriram Selvam, Anneswa Ghosh  
**Contact**: For questions about the dataset or pipeline, please open an issue on GitHub.