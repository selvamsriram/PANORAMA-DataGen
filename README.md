# File description

### DataGeneration/Project/WikiSeedDataGeneration.py
Reads `wikimedia/structured-wikipedia` from HuggingFace and extracts the following fields
 - Name
 - URL
 - Abstract
 - Wikipedia_Content
 - Personal_Life
 - Early_Life

Saves it to `DataGeneration/Data/wikipedia_people_{timestamp}.tsv`
Same is converted to JSONL here `DataGeneration/Data/wikipedia_people_{timestamp}.jsonl`

### DataGeneration/Project/SyntheticProfileGenerator.py
A constrained synthetic data generator that uses Faker library to create data for following locales
```
'en_US', 'en_GB', 'en_CA', 'en_AU', 'en_NZ', 'en_IE', 'en_IN', 'en_PH'
```
Saves created data to `DataGeneration/Data/synthetic_profiles_{timestamp}.tsv`

### DataGeneration/Project/PromptSyntheticDataGenTemplatedApproachV1.md
Step 1 of synthetic data generation.
Takes the following inputs,

 - synthetic_profile_json generated in `SyntheticProfileGenerator.py`
 - real_wiki_inspiration_text generated in `WikiSeedDataGeneration.py`

Produces a synthetic wikipedia style article

### DataGeneration/Project/SubmitToAzureFoundry.py
Uses the `PromptSyntheticDataGenTemplatedApproachV1.md` and submits it to Azure AI Foundry O3-Mini endpoint to generate the synthetic article on a person entity.

Results saved in `DataGeneration/Data/azure_batch_job_{timestamp}.live.jsonl`

Aggregated results are in `DataGeneration/Data/generated_passages_azure_results.10K.jsonl`

### DataGeneration/Project/ExtractGeneratedSyntheticPassage.py
The generation prompt produces the following

 - [Synthetic Article]
 - [Real Person Text Usage Notes]
 - [Synthetic Profile Usage Notes]

Extract the [Synthetic Article] for the next step here

Results stored in `DataGeneration/Data/extracted_passages_azure_results.10K.jsonl`

### DataGeneration/Project/SyntheticSocialDataGenPrompt.md
Prompt that consumes `Synthetic Persona Article` and `Synthetic Profile Json` and produces a synthetic data in the following formats

- Social Media
- Forum Posts
- Online Review
- Blog/News Article Comment
- Online Marketplace/Classified Ad Listing

### DataGeneration/Project/AzureGenerateSyntheticTrainingData.py
Use the `SyntheticSocialDataGenPrompt.md` along with `extracted_passages_azure_results.10K.jsonl` to generate the synthetic variety data in `Azure_Synthetic_Data_10K.{split}_{timestamp}.live_output.jsonl.`

Aggregated results are in `DataGeneration/Data/Azure_Synthetic_Data_10K.combined.jsonl`

### DataGeneration/Project/process_synthetic_data.py
Extracts the various synthetic data generated into `DataGeneration/Data/Azure_Synthetic_Data_10K.processed.jsonl`.
Also creates the HuggingFace Dataset at `DataGeneration/Data/Azure_Synthetic_Data_10K.pretraining.hf.tsv`