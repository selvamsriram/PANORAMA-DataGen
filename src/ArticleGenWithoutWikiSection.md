# Role:
You are a synthetic Wikipedia-style article creator who uses the provided input data to generate articles that can be used for research on Language Models.

# Objective:
Our goal is to create an article based on the provided synthetic persona and make it as realistic as possible.
You will be provided an input synthetic_profile_json, which contains a synthetic profile with all the essential details such as name, parents, job, education, etc.

# How to use data from synthetic_profile_json:

All the factual details for this article must come from the synthetic_profile_json.
You must only use the aspects from the synthetic profile that are required to create the article.
It is okay to leave some fields from the synthetic profile unused.

However, all added nuance must be plausible extrapolation from the synthetic profile, not sourced from any real person's data.

# Mandatory Constraints:

Content Source: 100% of the factual information in the output must originate from the synthetic_profile_json.
Format: Output should be a well-structured text passage with as many sections as needed.

Output Format:
[Synthetic Article][Generated Text]
[Synthetic Profile Usage Notes][Details from the synthetic profile that have been used]

# Inputs:
synthetic_profile_json: 
{
 "Unique ID": "176a9d59-cae8-4b80-8e87-7b125b95f93e",
 "Locale": "en_US",
 "First Name": "Rebecca",
 "Last Name": "Flores",
 "Father's Name": "John Flores",
 "Mother's Name": "Karen Flores",
 "Gender": "Female",
 "Age": 29,
 "Nationality": "American",
 "Marital Status": "Single",
 "Spouse Name": "N/A",
 "Children Count": 0,
 "National ID": "019-42-5061",
 "Passport Number": "N/A",
 "Driver's License": "US-DL-39898836644",
 "Phone Number": "391-270-7071x21889",
 "Work Phone": "(270)849-0436x40583",
 "Address": "0667 Jessica Port Suite 462, Smithhaven, HI 04247",
 "Email Address": "rebecca.flores38@aol.com",
 "Work Email": "r.flores@christian.info",
 "Birth Date": "1995-12-25",
 "Birth City": "West Robert",
 "Education Info": "Bachelor's",
 "Finance Status": "Low",
 "Net Worth": "$31473.09",
 "Employer": "Christian LLC",
 "Job Title": "Biologist",
 "Annual Salary": "$115004.68",
 "Credit Score": 572,
 "Social Media Handles": "{'Facebook': '@rebecca959', 'Instagram': '@floresr'}",
 "Blood Type": "B+",
 "Allergies": "Soy",
 "Disability": "None",
 "Emergency Contact Name": "Howard Kelley",
 "Emergency Contact Phone": "001-801-216-6079x52737"
}