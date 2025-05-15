**Role:** You are an expert biographical writer and meticulous researcher, specializing in creating neutral, encyclopedic narratives in the style of Wikipedia articles.

**Objective:** Generate a biographical passage about a synthetic individual. The factual basis for this passage MUST come *exclusively* from the provided `synthetic_profile_json`. The writing style, tone, structure, and phrasing should emulate the provided `real_wiki_inspiration_text`, but **NO factual content** from the inspiration text should appear in the output.

**Inputs:**

1.  **Synthetic Profile Data (`synthetic_profile_json`):**
    ```json
    {synthetic_profile_json}
    ```
    * This JSON object contains all the permissible factual information (names, dates, locations, career details, education, relationships, etc.) for the biography.
    * Treat this as the **sole source of truth** for the individual's life story.

2.  **Real Wikipedia Biography (`real_wiki_inspiration_text`):**
    ```text
    {real_wiki_inspiration_text}
    ```
    * Purpose: **STRICTLY for stylistic and structural inspiration.** Analyze this text to understand:
        * **Tone:** Neutral, objective, factual, encyclopedic.
        * **Sentence Structure:** Varied, clear, declarative.
        * **Flow & Transitions:** How different life phases or achievements are connected.
        * **Common Phrasing:** Typical biographical language (e.g., "Born in...", "Graduated from...", "Began their career as...", "Is known for...").
        * **Organization:** Typical sections or logical progression (e.g., Early Life, Career, Personal Life, Legacy - adapt based on available synthetic data).
    * **CRITICAL:** You MUST **ignore and discard all specific factual content** (names, dates, places, events, achievements, relationships, quotes, etc.) from this inspiration text. It serves *only* as a template for *how* to write, not *what* to write about.

**Detailed Instructions:**

1.  **Analyze Stylistic Inspiration:** Read the `real_wiki_inspiration_text` carefully to internalize its encyclopedic style, neutral tone, structure, and common biographical phrasing patterns. **Immediately after this analysis, purge all memory of the specific facts contained within it.**
2.  **Parse Synthetic Data:** Thoroughly examine the `synthetic_profile_json`. Identify all available biographical data points relevant to a life story.
3.  **Synthesize Narrative:** Construct a biographical passage based *only* on the data from `synthetic_profile_json`.
    * **Integrate Naturally:** Weave the JSON data points into a smooth, coherent narrative. Do not simply list facts. Create connections between different pieces of information (e.g., link education to career start, location changes to life events). Use as much of the available JSON data as possible without making the text feel forced or unnatural.
    * **Emulate Style:** Write the passage using the neutral, objective tone and encyclopedic sentence structures observed in the `real_wiki_inspiration_text`. Use similar transition words and biographical phrasing where appropriate, but apply them to the *synthetic* facts.
    * **Logical Structure:** Organize the information logically. A chronological approach (early life, education, career, later life/personal details) is often best, but adapt based on the available data in the JSON. Use paragraphs to separate distinct phases or aspects of the synthetic individual's life.
    * **Natural Usage** Though there are many fields like income, etc are provided in the synthetic profile it doesn't mean they can be used directly, this data can be transformed in a naturally reasonable manner and be used. The result should resemble natural text found in wiki and shouldn't include things that are too obviously glaring to be present in a neutral online profile.

**Mandatory Constraints:**

* **Content Source:** 100% of the factual information in the output *must* originate from the `synthetic_profile_json`.
* **Inspiration Content Exclusion:** **ZERO** specific facts (names, dates, locations, job titles, company names, events, relationships, achievements, numbers, quotes, etc.) from the `real_wiki_inspiration_text` are allowed in the output. Cross-contamination is a critical failure.
* **Tone:** Strictly neutral and encyclopedic.
* **Format:** Output should be a well-structured text passage with as many sections as needed

**Output:**

