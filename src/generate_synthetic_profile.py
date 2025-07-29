from datetime import datetime, date
import random
import csv
import uuid
from faker import Faker
import math # Added for floor function
import warnings # To warn about unmapped jobs
import re # For cleaning employer names for email

# --- Locales and Associated Data ---
locales = [
    'en_US', 'en_GB', 'en_CA', 'en_AU', 'en_NZ', 'en_IE', 'en_IN', 'en_PH'
]

# Currency symbols based on locale
currency_symbols = {
    "en_US": "$", "en_GB": "£", "en_CA": "$", "en_AU": "$", "en_NZ": "$",
    "en_IE": "€", "en_IN": "₹", "en_PH": "₱", "en_SG": "S$", "en_ZA": "R"
}

# International dialing codes based on locale
country_codes = {
    "en_US": "+1", "en_GB": "+44", "en_CA": "+1", "en_AU": "+61", "en_NZ": "+64",
    "en_IE": "+353", "en_IN": "+91", "en_PH": "+63", "en_SG": "+65", "en_ZA": "+27"
}

# --- Categorical Lists and Weights ---

genders = ["Male", "Female", "Non-Binary", "Other"]
gender_weights = [48, 48, 3, 1]

marital_statuses = ["Single", "Married", "Divorced", "Widowed"]
marital_status_weights = [35, 50, 10, 5] # Base weights, adjusted by age later

education_levels = [
    "Less than High School", "High School", "Some College", "Associate's",
    "Vocational Training", "Diploma", "Professional Certificate", "Bachelor's",
    "Master's", "PhD"
]
education_weights = [5, 25, 15, 10, 5, 5, 5, 25, 10, 3] # Base weights

finance_statuses = ["Very Low", "Low", "Medium", "High", "Very High"]
finance_weights = [5, 20, 40, 25, 10] # Base weights, adjusted by salary/locale later

# --- Job Positions and Weights ---
# (Ensure this list is comprehensive and matches weights)
expanded_job_positions = [ "Software Engineer", "Doctor", "Teacher", "Financial Analyst", "Sales Manager", "Consultant", "Nurse", "Project Manager", "Accountant", "Mechanic", "Lawyer", "Marketing Manager", "HR Specialist", "Customer Service Representative", "Scientist", "Researcher", "Architect", "Retail Sales Associate", "Cashier", "Waiter/Waitress", "Cook", "Administrative Assistant", "Truck Driver", "General Laborer", "Office Manager", "Registered Nurse (RN)", "Nursing Assistant", "Home Health Aide", "Medical Assistant", "Electrician", "Plumber", "Carpenter", "Construction Worker", "Warehouse Worker", "Delivery Driver", "Police Officer", "Firefighter", "Paramedic/EMT", "Graphic Designer", "Web Developer", "Data Scientist", "Data Analyst", "IT Support Specialist", "Network Engineer", "Systems Administrator", "Financial Advisor", "Insurance Agent", "Real Estate Agent", "Marketing Specialist", "Social Media Manager", "Human Resources Manager", "Recruiter", "Paralegal", "Librarian", "Pharmacist", "Pharmacy Technician", "Physical Therapist", "Occupational Therapist", "Dental Hygienist", "School Principal", "Professor/University Lecturer", "Chef", "Bartender", "Hotel Desk Clerk", "Cleaner/Janitor", "Security Guard", "Farmer/Agricultural Worker", "Machinist", "Welder", "HVAC Technician", "Bus Driver", "Pilot", "Air Traffic Controller", "Journalist", "Writer/Author", "Editor", "Photographer", "Social Worker", "Psychologist/Therapist", "Engineer (Civil)", "Engineer (Mechanical)", "Engineer (Electrical)", "Biologist", "Chemist", "Physicist", "Geologist", "Environmental Scientist", "Veterinarian", "Veterinary Technician", "Fitness Trainer", "Hair Stylist/Barber", "Childcare Worker", "Event Planner", "Translator/Interpreter", "Executive Assistant", "Operations Manager", "Supply Chain Manager", "Auditor", "Bookkeeper", "Loan Officer", "Underwriter" ]
expanded_job_weights = [ 15, 8, 18, 8, 9, 6, 25, 10, 12, 10, 7, 8, 8, 25, 5, 4, 5, 40, 35, 28, 26, 30, 22, 20, 12, 20, 15, 16, 14, 10, 9, 9, 15, 18, 17, 8, 6, 7, 7, 12, 6, 9, 13, 7, 8, 7, 7, 6, 10, 6, 7, 7, 7, 4, 7, 9, 6, 5, 6, 5, 5, 7, 9, 10, 25, 15, 8, 6, 7, 7, 8, 3, 2, 4, 5, 5, 6, 8, 7, 6, 7, 6, 4, 4, 3, 3, 4, 4, 5, 7, 10, 15, 4, 4, 8, 11, 6, 6, 10, 7, 5 ]
assert len(expanded_job_positions) == len(expanded_job_weights), "Job positions and weights lists must have the same length."

# --- Job Tiers, Salary Ranges (Base USD Estimates), and Locale Multipliers ---
# These ranges are VERY rough estimates for a US baseline.
# Locale multipliers are EXTREMELY simplified examples of cost-of-living adjustments.
JOB_TIERS = {
    "Entry/Service": {"min": 15000, "max": 45000, "titles": ["Retail Sales Associate", "Cashier", "Waiter/Waitress", "Cook", "Administrative Assistant", "Customer Service Representative", "Nursing Assistant", "Home Health Aide", "Medical Assistant", "Warehouse Worker", "Delivery Driver", "Cleaner/Janitor", "Security Guard", "Pharmacy Technician", "Hotel Desk Clerk", "Bartender", "Childcare Worker", "Hair Stylist/Barber", "Fitness Trainer", "Bookkeeper"]},
    "Skilled Trades/Technical": {"min": 40000, "max": 85000, "titles": ["Mechanic", "Electrician", "Plumber", "Carpenter", "Construction Worker", "General Laborer", "IT Support Specialist", "Machinist", "Welder", "HVAC Technician", "Bus Driver", "Veterinary Technician", "Vocational Training", "Diploma", "Professional Certificate", "Paralegal"]},
    "Professional/Analyst": {"min": 50000, "max": 120000, "titles": ["Software Engineer", "Teacher", "Financial Analyst", "Nurse", "Registered Nurse (RN)", "Accountant", "Marketing Specialist", "HR Specialist", "Graphic Designer", "Web Developer", "Data Analyst", "Network Engineer", "Systems Administrator", "Financial Advisor", "Insurance Agent", "Real Estate Agent", "Recruiter", "Librarian", "Physical Therapist", "Occupational Therapist", "Dental Hygienist", "Social Worker", "Journalist", "Writer/Author", "Editor", "Photographer", "Translator/Interpreter", "Auditor", "Loan Officer"]},
    "High Professional/Specialized": {"min": 70000, "max": 250000, "titles": ["Doctor", "Lawyer", "Scientist", "Researcher", "Architect", "Data Scientist", "Pharmacist", "Professor/University Lecturer", "Chef", "Pilot", "Air Traffic Controller", "Psychologist/Therapist", "Engineer (Civil)", "Engineer (Mechanical)", "Engineer (Electrical)", "Biologist", "Chemist", "Physicist", "Geologist", "Environmental Scientist", "Veterinarian", "Underwriter"]},
    "Management/Executive": {"min": 65000, "max": 300000, "titles": ["Sales Manager", "Consultant", "Project Manager", "Marketing Manager", "Office Manager", "Human Resources Manager", "School Principal", "Executive Assistant", "Operations Manager", "Supply Chain Manager", "Farmer/Agricultural Worker"]}
}

# Basic Cost-of-Living Multiplier Examples (1.0 = US Baseline)
# These are illustrative ONLY and not precise economic figures.
LOCALE_SALARY_MULTIPLIERS = {
    'en_US': 1.0, 'en_GB': 0.9, 'en_CA': 0.95, 'en_AU': 1.0, 'en_NZ': 0.85,
    'en_IE': 0.9, 'en_IN': 0.3, 'en_PH': 0.25, 'en_SG': 1.1, 'en_ZA': 0.4,
    'default': 0.7 # Default for any unlisted locales
}

# --- Verification: Check Job Tier Coverage ---
all_tiered_jobs = set()
for tier_info in JOB_TIERS.values():
    all_tiered_jobs.update(tier_info["titles"])
unmapped_jobs = set(expanded_job_positions) - all_tiered_jobs
if unmapped_jobs:
    warnings.warn(f"The following job titles are not mapped in JOB_TIERS and will use default salary/requirements: {unmapped_jobs}")

# Create a reverse lookup: job_title -> tier_info
JOB_TITLE_TO_TIER = {}
for tier_name, tier_info in JOB_TIERS.items():
    for title in tier_info["titles"]:
        JOB_TITLE_TO_TIER[title] = {"tier": tier_name, "min": tier_info["min"], "max": tier_info["max"]}

# --- Other Categorical Data ---
social_platforms = [ "Twitter", "Facebook", "LinkedIn", "Instagram", "TikTok", "Snapchat", "Reddit", "Pinterest" ]
social_platform_weights = [15, 20, 10, 20, 15, 10, 10, 5]

# Nationalities - Base list, will be biased by locale during generation
nationalities = [ "American", "British", "Indian", "Australian", "Canadian", "Chinese", "German", "French", "Italian", "Spanish", "Mexican", "Brazilian", "Japanese", "Russian", "South African", "Filipino", "Irish", "New Zealander", "Singaporean" ] # Added more to match locales
nationality_locale_map = { # Mapping locale code to likely nationality
    'en_US': 'American', 'en_GB': 'British', 'en_CA': 'Canadian', 'en_AU': 'Australian',
    'en_NZ': 'New Zealander', 'en_IE': 'Irish', 'en_IN': 'Indian', 'en_PH': 'Filipino',
    'en_SG': 'Singaporean', 'en_ZA': 'South African'
}

blood_types = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
blood_type_weights = [34, 6, 9, 2, 3, 1, 37, 7]

allergy_options = [ "None", "Pollen", "Dust", "Gluten", "Peanuts", "Seafood", "Tree Nuts", "Soy", "Eggs", "Milk", "Latex", "Insect Stings" ]
allergy_weights = [70, 10, 10, 5, 5, 5, 3, 2, 2, 2, 1, 1]

disability_statuses = [ "None", "Visual", "Hearing", "Mobility", "Cognitive", "Speech", "Chronic Illness", "Mental Health" ]
disability_weights = [85, 5, 5, 3, 1, 1, 3, 2]


# --- Helper Functions ---

def generate_national_id(fake, locale):
    """Generates a national ID, using US SSN for en_US and placeholders otherwise."""
    if locale == 'en_US':
        try:
            return fake.ssn()
        except AttributeError: # Fallback if locale provider doesn't have ssn
             return f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}"
    # Placeholder for other locales - replace with specific formats if Faker supports them
    # e.g., fake.nin() for UK, fake.tfn() for AU etc.
    else:
        # Generic placeholder format
        prefix = locale.split('_')[1] if '_' in locale else locale.upper()
        return f"{prefix}-ID-{random.randint(1000000, 9999999)}"

def generate_realistic_email(first_name, last_name, locale, fake):
    """Generates a realistic email address using locale-specific domains."""
    email_domains = { "en_US": ["gmail.com", "yahoo.com", "outlook.com", "aol.com", "icloud.com", "hotmail.com", "live.com", "protonmail.com", "mail.com"], "en_GB": ["gmail.com", "yahoo.co.uk", "hotmail.co.uk", "outlook.com", "icloud.com", "btinternet.com", "live.co.uk", "protonmail.com"], "en_CA": ["gmail.com", "yahoo.ca", "outlook.com", "icloud.com", "bell.net", "rogers.com", "shaw.ca", "sympatico.ca"], "en_AU": ["gmail.com", "yahoo.com.au", "outlook.com", "bigpond.com", "icloud.com", "live.com.au", "optusnet.com.au"], "en_NZ": ["gmail.com", "outlook.com", "xtra.co.nz", "vodafone.co.nz", "yahoo.co.nz", "hotmail.com"], "en_IE": ["gmail.com", "yahoo.ie", "outlook.com", "eircom.net", "hotmail.ie", "live.ie"], "en_IN": ["gmail.com", "yahoo.in", "rediffmail.com", "outlook.com", "hotmail.com", "protonmail.com", "live.in"], "en_PH": ["gmail.com", "yahoo.com.ph", "outlook.ph", "icloud.com", "rocketmail.com", "hotmail.ph"], "en_SG": ["gmail.com", "yahoo.com.sg", "singnet.com.sg", "outlook.com", "icloud.com", "hotmail.sg"], "en_ZA": ["gmail.com", "yahoo.co.za", "mweb.co.za", "outlook.com", "icloud.com", "hotmail.co.za", "webmail.co.za"] }
    domain = random.choice(email_domains.get(locale, ["gmail.com", fake.free_email_domain()])) # Use faker domain as fallback

    # Clean names for email generation (remove spaces, common titles)
    clean_first = first_name.lower().split(' ')[0]
    clean_last = last_name.lower().split(' ')[-1] # Use last part of last name

    # Use faker's user_name for more variety sometimes
    if random.random() < 0.2:
        username = fake.user_name()
    else:
        # Generate email username in various common formats
        formats = [
            f"{clean_first}.{clean_last}",
            f"{clean_first}{clean_last}",
            f"{clean_first[0]}{clean_last}",
            f"{clean_first}_{clean_last}",
            f"{clean_last}{random.randint(1, 99)}",
            f"{clean_first}{random.randint(100, 999)}",
            f"{clean_first}.{clean_last}{random.randint(1, 99)}"
        ]
        username = random.choice(formats)
        # Remove potential invalid characters just in case
        username = re.sub(r'[^a-z0-9._]', '', username)

    return f"{username}@{domain}"

def generate_passport_number(fake):
    """Generates a generic passport number."""
    try:
        return fake.passport_number()
    except AttributeError:
        letters = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=random.randint(1, 2)))
        digits = ''.join([str(random.randint(0, 9)) for _ in range(random.randint(7, 9))])
        return letters + digits

def generate_drivers_license(fake, locale):
    """Generates a driver's license number, attempting locale specificity."""
    try:
        # Placeholder: Use a generic format if no specific method found
        prefix = locale.split('_')[1] if '_' in locale else locale.upper()
        digits = ''.join([str(random.randint(0, 9)) for _ in range(random.randint(7, 12))])
        return f"{prefix}-DL-{digits}"
    except AttributeError: # Fallback if no provider method exists
        letter = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        digits = ''.join([str(random.randint(0, 9)) for _ in range(8)])
        return letter + digits

# --- NEW: Helper Function for Social Media Handles ---
def generate_social_handle(first_name, last_name, platform, fake):
    """Generates a more realistic social media handle based on the person's name."""
    # Clean names (lowercase, remove spaces)
    fn_clean = re.sub(r'\s+', '', first_name.lower())
    ln_clean = re.sub(r'\s+', '', last_name.lower())
    fn_initial = fn_clean[0] if fn_clean else ''

    # Possible formats
    formats = [
        f"{fn_clean}{ln_clean}",
        f"{fn_clean}.{ln_clean}",
        f"{fn_clean}_{ln_clean}",
        f"{fn_initial}{ln_clean}",
        f"{ln_clean}{fn_initial}",
        f"{fn_clean}{random.randint(1, 999)}",
        f"{ln_clean}{random.randint(1, 999)}",
        f"{fn_clean}_{random.randint(1, 99)}",
        f"{ln_clean}_{random.randint(1, 99)}",
    ]

    # Platform specific tweaks (optional)
    if platform == "LinkedIn": # Tend towards more professional formats
        pro_formats = [f"{fn_clean}{ln_clean}", f"{fn_clean}.{ln_clean}", f"{fn_initial}{ln_clean}"]
        if random.random() < 0.7: # High chance of professional format
            handle = random.choice(pro_formats)
        else:
            handle = random.choice(formats)
    # elif platform == "Twitter" or platform == "TikTok": # Often shorter, sometimes more abstract
    #      if random.random() < 0.3:
    #          handle = fake.user_name() # Keep some purely random ones
    #      else:
    #         handle = random.choice(formats)
    else: # Default for Facebook, Instagram etc.
        handle = random.choice(formats)

    # Basic length limiting (example)
    handle = handle[:20] # Limit length for realism

    # Ensure it starts with @
    return f"@{handle}"


# --- Main Data Generation Function ---

def generate_person(fake, locale, currency_symbol, country_code):
    """Generates a single synthetic person record with internal consistency."""

    # 1. Generate Base Demographics & Names
    gender = random.choices(genders, weights=gender_weights, k=1)[0]
    if gender == "Male":
        first_name = fake.first_name_male()
        mother_first_name = fake.first_name_female()
        father_first_name = fake.first_name_male()
    elif gender == "Female":
        first_name = fake.first_name_female()
        mother_first_name = fake.first_name_female()
        father_first_name = fake.first_name_male()
    else: # Non-Binary / Other
        first_name = fake.first_name() # Use generic name
        mother_first_name = fake.first_name_female()
        father_first_name = fake.first_name_male()

    # Generate distinct last names for parents vs person's birth name
    primary_last_name = fake.last_name() # Person's original last name is also the family last name at this point
    father_last_name = primary_last_name # Father's last name (could be different)
    # Assume mother took father's last name for this simulation
    mother_last_name = primary_last_name

    # Assign parent names
    father_name = f"{father_first_name} {father_last_name}"
    mother_name = f"{mother_first_name} {mother_last_name}"

    # Initialize person's current last name to their birth name
    current_last_name = primary_last_name

    # Generate birth date and calculate age
    birth_date_obj = fake.date_of_birth(minimum_age=18, maximum_age=90)
    birth_date = birth_date_obj.strftime("%Y-%m-%d")
    today = date.today()
    age = today.year - birth_date_obj.year - ((today.month, today.day) < (birth_date_obj.month, birth_date_obj.day))

    # 2. Age-Constrained Education
    possible_educations = []
    possible_education_weights = []
    min_age_for_edu = { "Less than High School": 0, "High School": 16, "Some College": 18, "Associate's": 19, "Vocational Training": 18, "Diploma": 18, "Professional Certificate": 18, "Bachelor's": 21, "Master's": 22, "PhD": 25 }
    for i, edu in enumerate(education_levels):
        if age >= min_age_for_edu.get(edu, 0):
            possible_educations.append(edu)
            possible_education_weights.append(education_weights[i])
    if not possible_educations: possible_educations, possible_education_weights = ["High School"], [1]
    education = random.choices(possible_educations, weights=possible_education_weights, k=1)[0]

    # 3. Age/Education-Constrained Marital Status & Children (Update Last Name Logic)
    current_marital_weights = list(marital_status_weights)
    if age < 22: current_marital_weights[1]*=0.2; current_marital_weights[2]*=0.1; current_marital_weights[3]*=0.05; current_marital_weights[0]*=1.5
    elif age > 65: current_marital_weights[3]*=3; current_marital_weights[0]*=0.5
    marital_status = random.choices(marital_statuses, weights=[max(1, w) for w in current_marital_weights], k=1)[0]

    spouse_name = "N/A"
    spouse_last_name = "N/A" # Store spouse's last name separately
    historical_spouse_suffix = ""
    generate_base_spouse = False

    if marital_status == "Married":
        generate_base_spouse = True
    elif marital_status == "Widowed":
        historical_spouse_suffix = " (Deceased)"
        generate_base_spouse = random.random() < 0.7
    elif marital_status == "Divorced":
        historical_spouse_suffix = " (Ex)"
        generate_base_spouse = random.random() < 0.6

    if generate_base_spouse:
         spouse_first = fake.first_name()
         spouse_last_name = fake.last_name() # Generate spouse's distinct last name
         spouse_name = f"{spouse_first} {spouse_last_name}{historical_spouse_suffix}"

         # --- Update Person's Last Name (if Female & Married) ---
         # Check if the person is female and is currently married (not just historically)
         if gender == "Female" and marital_status == "Married":
             # Probability of taking spouse's name (e.g., 80%)
             if random.random() < 0.80:
                 current_last_name = spouse_last_name # Update the main last name variable

    # Children count based on status and age
    children_count = 0
    if marital_status == "Married" and age > 20:
        children_count = random.choices([0, 1, 2, 3, 4], weights=[20, 30, 30, 15, 5], k=1)[0]
    elif marital_status in ["Widowed", "Divorced"] and age > 22:
        children_count = random.choices([0, 1, 2, 3, 4], weights=[30, 30, 25, 10, 5], k=1)[0]
    if age < 20: children_count = 0

    # --- Now use `current_last_name` for subsequent fields like email, social handles ---

    # 4. Education/Age-Constrained Job Title (Stricter Logic - No changes needed here from v3.1)
    possible_jobs = []
    possible_job_weights = []
    min_education_level_map = {edu: i for i, edu in enumerate(education_levels)}
    current_edu_level = min_education_level_map.get(education, 0)
    for i, job in enumerate(expanded_job_positions):
        job_info = JOB_TITLE_TO_TIER.get(job)
        required_edu_level = 0; min_job_age = 18
        if job_info:
            tier = job_info["tier"]
            if tier in ["Professional/Analyst", "Management/Executive"]: required_edu_level = min_education_level_map.get("Associate's", 3)
            if tier in ["High Professional/Specialized"]: required_edu_level = min_education_level_map.get("Bachelor's", 7)
            if tier == "Management/Executive": min_job_age = 25
            if job in ["Doctor", "Lawyer", "PhD", "Professor/University Lecturer", "Scientist"]: required_edu_level = min_education_level_map.get("Master's", 8); min_job_age = 25
            if job in ["Pilot", "Police Officer", "Air Traffic Controller"]: min_job_age = 21
        if age >= min_job_age and current_edu_level >= required_edu_level:
            weight_multiplier = 1.0
            education_gap = current_edu_level - required_edu_level
            if education_gap >= 5: weight_multiplier = 0.01
            elif education_gap >= 3: weight_multiplier = 0.2
            possible_jobs.append(job)
            possible_job_weights.append(max(0.01, expanded_job_weights[i] * weight_multiplier))
    if not possible_jobs:
        warnings.warn(f"No suitable jobs found for Age={age}, Education='{education}'. Using fallback.")
        possible_jobs = ["Administrative Assistant", "Customer Service Representative", "Office Manager"]
        possible_job_weights = [1, 1, 1]
    final_weights = [max(0.1, w) for w in possible_job_weights]
    job_title = random.choices(possible_jobs, weights=final_weights, k=1)[0]

    # 5. Job-Based Salary with Locale Adjustment (No changes needed here from v3.1)
    job_tier_info = JOB_TITLE_TO_TIER.get(job_title)
    locale_multiplier = LOCALE_SALARY_MULTIPLIERS.get(locale, LOCALE_SALARY_MULTIPLIERS['default'])
    base_salary_min = 20000; base_salary_max = 60000
    if job_tier_info: base_salary_min = job_tier_info["min"]; base_salary_max = job_tier_info["max"]
    elif job_title in unmapped_jobs: base_salary_min = 35000; base_salary_max = 75000
    salary_min = base_salary_min * locale_multiplier; salary_max = base_salary_max * locale_multiplier
    experience_factor = min(1.5, 1 + (age - 18) / 40.0)
    salary_min_adj = int(salary_min * (0.8 + random.random() * 0.4))
    salary_max_adj = int(salary_max * (0.8 + random.random() * 0.4))
    if salary_max_adj <= salary_min_adj : salary_max_adj = salary_min_adj + 1000 * locale_multiplier
    salary_value = round(random.uniform(salary_min_adj, salary_max_adj) * experience_factor, 2)
    min_locale_salary = 10000 * locale_multiplier
    if age < 21: salary_value = min(salary_value, 60000 * locale_multiplier * (1 + random.random() * 0.2))
    salary = f"{currency_symbol}{max(min_locale_salary, salary_value)}"

    # 6. Salary/Age/Locale-Based Finance Status & Net Worth (No changes needed here from v3.1)
    current_finance_weights = list(finance_weights)
    salary_percentile = (salary_value - min_locale_salary) / max(1, (300000 * locale_multiplier) - min_locale_salary)
    salary_percentile = max(0, min(1, salary_percentile))
    if salary_percentile < 0.15: current_finance_weights[0]*=2.0; current_finance_weights[1]*=1.5; current_finance_weights[3]*=0.5; current_finance_weights[4]*=0.2
    elif salary_percentile < 0.4: current_finance_weights[1]*=1.5; current_finance_weights[2]*=1.2; current_finance_weights[4]*=0.5
    elif salary_percentile > 0.75: current_finance_weights[3]*=1.5; current_finance_weights[4]*=2.0; current_finance_weights[0]*=0.3; current_finance_weights[1]*=0.5
    finance_status = random.choices(finance_statuses, weights=[max(0.1, w) for w in current_finance_weights], k=1)[0]
    net_worth_multiplier = { "Very Low": 0.1, "Low": 0.5, "Medium": 1.0, "High": 2.5, "Very High": 5.0 }.get(finance_status, 1.0)
    years_working = max(0, age - 18)
    estimated_savings_rate = random.uniform(0.02, 0.15)
    base_net_worth = (salary_value * estimated_savings_rate) * years_working * net_worth_multiplier
    net_worth_value = round(max(0, base_net_worth * random.uniform(0.5, 1.5)) + random.uniform(-5000, 10000) * locale_multiplier, 2)
    if age < 30 and random.random() < 0.2: net_worth_value = max(net_worth_value, random.uniform(-20000 * locale_multiplier, 5000 * locale_multiplier))
    net_worth = f"{currency_symbol}{net_worth_value}"

    # 7. Nationality (Biased towards Locale - No changes needed here from v3.1)
    nationality = nationality_locale_map.get(locale)
    # nationality = "N/A"
    # if target_nationality:
    #     if random.random() < 0.9: nationality = target_nationality
    #     else:
    #         possible_others = [n for n in nationalities if n != target_nationality]
    #         if possible_others: nationality = random.choice(possible_others)
    #         else: nationality = target_nationality

    # 8. Other Fields (Generate remaining details)

    # Generate Emails *after* potential last name change
    # Use current_last_name for personal email
    email = generate_realistic_email(first_name, current_last_name, locale, fake)

    # Employer and Work Email
    try: employer = fake.company() if job_title not in ["General Laborer", "Retail Sales Associate"] else fake.company_suffix() + " Store"
    except AttributeError: employer = "Unknown Employer"
    work_email = fake.company_email() # Default
    if job_info and job_info['tier'] != "Entry/Service" and isinstance(employer, str) and len(employer) > 3:
        try:
            employer_domain_part = employer.lower().split(' ')[0].split(',')[0].split('.')[0]
            employer_domain_part = ''.join(filter(str.isalnum, employer_domain_part))
            if len(employer_domain_part) > 3 and employer_domain_part not in ["store", "shop", "ltd", "inc", "llc", "corp", "group"]:
                 # Use current_last_name for work email as well
                 work_email = f"{first_name[0].lower()}.{current_last_name.lower()}@{employer_domain_part}.{fake.tld()}"
        except Exception: pass # Stick with default

    # Phone numbers
    try: phone = fake.phone_number()
    except AttributeError: phone = f"{country_code} {random.randint(100000000, 999999999)}"
    work_phone = "N/A"
    if random.random() < 0.7:
        try: work_phone = fake.phone_number()
        except AttributeError: work_phone = f"{country_code} {random.randint(100000000, 999999999)} (Work)"

    # Address and Birth City
    address = fake.address().replace('\n', ', ')
    try: birth_city = fake.city()
    except AttributeError: birth_city = "Unknown City"

    # Credit Score
    credit_score_base = { "Very Low": 450, "Low": 550, "Medium": 650, "High": 720, "Very High": 760 }.get(finance_status, 600)
    credit_score_noise = random.randint(-80, 80)
    credit_score = max(300, min(850, credit_score_base + credit_score_noise + int((age - 18)*0.5) ))

    # Social Media Handles (using new helper and current_last_name)
    num_platforms = random.randint(0, 4)
    selected_platforms = random.choices( social_platforms, weights=social_platform_weights, k=num_platforms )
    social_media = {}
    if num_platforms > 0:
        # Use the potentially updated current_last_name for social handles
        social_media = {platform: generate_social_handle(first_name, current_last_name, platform, fake) for platform in selected_platforms}

    # IDs
    unique_id = str(uuid.uuid4())
    national_id = generate_national_id(fake, locale)
    passport = generate_passport_number(fake) if random.random() < 0.6 else "N/A"
    drivers_license = generate_drivers_license(fake, locale) if random.random() < 0.85 else "N/A"

    # Health Fields
    blood_type = random.choices(blood_types, weights=blood_type_weights, k=1)[0]
    allergies = random.choices(allergy_options, weights=allergy_weights, k=1)[0]
    disability = random.choices(disability_statuses, weights=disability_weights, k=1)[0]

    # Emergency Contact
    emergency_contact_name = fake.name()
    try: emergency_contact_phone = fake.phone_number()
    except AttributeError: emergency_contact_phone = f"{country_code} {random.randint(100000000, 999999999)}"


    # --- Return final record ---
    # Ensure the main "Last Name" field reflects the potentially updated name
    return {
        "Unique ID": unique_id,
        "Locale": locale,
        "First Name": first_name,
        "Last Name": current_last_name, # Use the potentially updated last name here
        "Father's Name": father_name, # Uses father_last_name
        "Mother's Name": mother_name, # Uses father_last_name (simulated)
        "Gender": gender,
        "Age": age,
        "Nationality": nationality,
        "Marital Status": marital_status,
        "Spouse Name": spouse_name, # Includes spouse's actual last name
        "Children Count": children_count,
        "National ID": national_id,
        "Passport Number": passport,
        "Driver's License": drivers_license,
        "Phone Number": phone,
        "Work Phone": work_phone,
        "Address": address,
        "Email Address": email, # Generated using current_last_name
        "Work Email": work_email, # Generated using current_last_name
        "Birth Date": birth_date,
        "Birth City": birth_city,
        "Education Info": education,
        "Finance Status": finance_status,
        "Net Worth": net_worth,
        "Employer": employer,
        "Job Title": job_title,
        "Annual Salary": salary,
        "Credit Score": credit_score,
        "Social Media Handles": str(social_media) if social_media else "{}", # Generated using current_last_name
        "Blood Type": blood_type,
        "Allergies": allergies,
        "Disability": disability,
        "Emergency Contact Name": emergency_contact_name,
        "Emergency Contact Phone": emergency_contact_phone
    }

# --- Main Execution Logic ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"/Users/sriramselvam/Code/PANORAMA-DataGen/data/synthetic_profiles_{timestamp}.tsv"

print(f"Generating synthetic dataset to: {output_file}")

# Define fieldnames (no changes needed here)
fieldnames = [
    "Unique ID", "Locale", "First Name", "Last Name", "Father's Name", "Mother's Name",
    "Gender", "Age", "Nationality", "Marital Status", "Spouse Name", "Children Count",
    "National ID", "Passport Number", "Driver's License", "Phone Number", "Work Phone",
    "Address", "Email Address", "Work Email", "Birth Date", "Birth City",
    "Education Info", "Finance Status", "Net Worth", "Employer", "Job Title",
    "Annual Salary", "Credit Score", "Social Media Handles", "Blood Type",
    "Allergies", "Disability", "Emergency Contact Name", "Emergency Contact Phone"
]

# Seed for reproducibility
Faker.seed(4321) # Consistent seed
random.seed(4321)

total_records_generated = 0
records_per_locale = 7500 # Keep reduced for testing, adjust as needed

try:
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()

        for locale in locales:
            print(f"Generating records for locale: {locale}...")
            try:
                fake = Faker(locale)
            except Exception as e:
                print(f"  Warning: Could not initialize Faker for locale '{locale}'. Skipping. Error: {e}")
                continue

            currency_symbol = currency_symbols.get(locale, "$")
            country_code = country_codes.get(locale, "+?")

            for i in range(records_per_locale):
                try:
                    record = generate_person(fake, locale, currency_symbol, country_code)
                    writer.writerow(record)
                    total_records_generated += 1
                except Exception as e:
                    print(f"\nError generating record {i+1} for locale {locale}: {e}")
                    import traceback
                    print(traceback.format_exc()) # Print full traceback for debugging errors
                    print("Attempting to continue...")


    print(f"\nSynthetic dataset generation complete. {total_records_generated} records written to '{output_file}'")

except IOError as e:
    print(f"\nError writing to file '{output_file}': {e}")
except Exception as e:
    print(f"\nAn unexpected error occurred during generation: {e}")
    import traceback
    print(traceback.format_exc())