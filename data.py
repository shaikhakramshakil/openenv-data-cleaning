"""
Synthetic dataset with planted data quality errors.

Contains realistic business data (customer records) with intentionally
planted errors for the agent to discover. Each error has a known ground
truth for deterministic grading.

Error types:
    - missing_value: Cell contains null/NaN/empty where data is expected
    - invalid_format: Value doesn't match expected format (email, date, etc.)
    - outlier: Value is statistically implausible (negative age, extreme amounts)
    - duplicate: Row is an exact or near-duplicate of another row
    - type_error: Value has wrong data type (string in numeric field)
    - inconsistency: Values within a row contradict each other
"""

from typing import Any, Dict, List, Tuple

# Column definitions with expected types and validation rules
COLUMNS = ["id", "name", "email", "age", "city", "signup_date", "plan", "monthly_amount", "status"]

COLUMN_DESCRIPTIONS = {
    "id": "Unique customer ID (integer, sequential)",
    "name": "Full name (string, 'First Last' format)",
    "email": "Email address (string, valid email format)",
    "age": "Customer age (integer, 18-100)",
    "city": "City of residence (string)",
    "signup_date": "Account signup date (string, YYYY-MM-DD format)",
    "plan": "Subscription plan (string, one of: free, basic, premium, enterprise)",
    "monthly_amount": "Monthly payment amount (float, must match plan pricing)",
    "status": "Account status (string, one of: active, inactive, suspended, cancelled)",
}

# Plan pricing rules (for inconsistency detection)
PLAN_PRICING = {
    "free": 0.00,
    "basic": 9.99,
    "premium": 29.99,
    "enterprise": 99.99,
}


def get_clean_dataset() -> List[Dict[str, Any]]:
    """Return the clean (error-free) version of the dataset."""
    return [
        {"id": 1,  "name": "Alice Johnson",    "email": "alice.johnson@gmail.com",    "age": 34, "city": "New York",      "signup_date": "2024-01-15", "plan": "premium",    "monthly_amount": 29.99, "status": "active"},
        {"id": 2,  "name": "Bob Smith",         "email": "bob.smith@yahoo.com",        "age": 28, "city": "Los Angeles",   "signup_date": "2024-02-20", "plan": "basic",      "monthly_amount": 9.99,  "status": "active"},
        {"id": 3,  "name": "Carol Williams",    "email": "carol.w@outlook.com",        "age": 45, "city": "Chicago",       "signup_date": "2024-03-10", "plan": "enterprise", "monthly_amount": 99.99, "status": "active"},
        {"id": 4,  "name": "David Brown",       "email": "david.brown@company.co",     "age": 52, "city": "Houston",       "signup_date": "2024-01-28", "plan": "free",       "monthly_amount": 0.00,  "status": "inactive"},
        {"id": 5,  "name": "Emma Davis",        "email": "emma.davis@email.com",       "age": 31, "city": "Phoenix",       "signup_date": "2024-04-05", "plan": "basic",      "monthly_amount": 9.99,  "status": "active"},
        {"id": 6,  "name": "Frank Miller",      "email": "frank.m@domain.org",         "age": 67, "city": "Philadelphia",  "signup_date": "2024-02-14", "plan": "premium",    "monthly_amount": 29.99, "status": "active"},
        {"id": 7,  "name": "Grace Wilson",      "email": "grace.wilson@mail.com",      "age": 23, "city": "San Antonio",   "signup_date": "2024-05-01", "plan": "free",       "monthly_amount": 0.00,  "status": "active"},
        {"id": 8,  "name": "Henry Taylor",      "email": "henry.t@inbox.com",          "age": 41, "city": "San Diego",     "signup_date": "2024-03-22", "plan": "basic",      "monthly_amount": 9.99,  "status": "cancelled"},
        {"id": 9,  "name": "Ivy Anderson",      "email": "ivy.anderson@webmail.com",   "age": 36, "city": "Dallas",        "signup_date": "2024-06-11", "plan": "enterprise", "monthly_amount": 99.99, "status": "active"},
        {"id": 10, "name": "Jack Thomas",       "email": "jack.thomas@service.net",    "age": 29, "city": "San Jose",      "signup_date": "2024-04-18", "plan": "premium",    "monthly_amount": 29.99, "status": "active"},
        {"id": 11, "name": "Karen Martinez",    "email": "karen.m@provider.com",       "age": 55, "city": "Austin",        "signup_date": "2024-01-07", "plan": "basic",      "monthly_amount": 9.99,  "status": "active"},
        {"id": 12, "name": "Leo Garcia",        "email": "leo.garcia@fastmail.com",    "age": 19, "city": "Jacksonville",  "signup_date": "2024-07-03", "plan": "free",       "monthly_amount": 0.00,  "status": "active"},
        {"id": 13, "name": "Mia Robinson",      "email": "mia.r@outlook.com",          "age": 48, "city": "Fort Worth",    "signup_date": "2024-02-28", "plan": "premium",    "monthly_amount": 29.99, "status": "suspended"},
        {"id": 14, "name": "Noah Clark",        "email": "noah.clark@gmail.com",       "age": 33, "city": "Columbus",      "signup_date": "2024-05-19", "plan": "enterprise", "monthly_amount": 99.99, "status": "active"},
        {"id": 15, "name": "Olivia Lewis",      "email": "olivia.l@yahoo.com",         "age": 27, "city": "Charlotte",     "signup_date": "2024-08-10", "plan": "basic",      "monthly_amount": 9.99,  "status": "active"},
    ]


def get_dirty_dataset() -> List[Dict[str, Any]]:
    """
    Return the dataset with planted errors.
    
    Errors are intentionally introduced to test agent data quality detection.
    Each error has a corresponding entry in get_ground_truth_errors().
    """
    return [
        # Row 1: CLEAN
        {"id": 1,  "name": "Alice Johnson",    "email": "alice.johnson@gmail.com",    "age": 34,   "city": "New York",      "signup_date": "2024-01-15", "plan": "premium",    "monthly_amount": 29.99, "status": "active"},
        # Row 2: ERROR - invalid email format (missing TLD)
        {"id": 2,  "name": "Bob Smith",         "email": "bob.smith@yahoo",            "age": 28,   "city": "Los Angeles",   "signup_date": "2024-02-20", "plan": "basic",      "monthly_amount": 9.99,  "status": "active"},
        # Row 3: ERROR - invalid date (Feb 30 doesn't exist)
        {"id": 3,  "name": "Carol Williams",    "email": "carol.w@outlook.com",        "age": 45,   "city": "Chicago",       "signup_date": "2024-02-30", "plan": "enterprise", "monthly_amount": 99.99, "status": "active"},
        # Row 4: CLEAN
        {"id": 4,  "name": "David Brown",       "email": "david.brown@company.co",     "age": 52,   "city": "Houston",       "signup_date": "2024-01-28", "plan": "free",       "monthly_amount": 0.00,  "status": "inactive"},
        # Row 5: ERROR - negative age (outlier)
        {"id": 5,  "name": "Emma Davis",        "email": "emma.davis@email.com",       "age": -3,   "city": "Phoenix",       "signup_date": "2024-04-05", "plan": "basic",      "monthly_amount": 9.99,  "status": "active"},
        # Row 6: ERROR - plan/amount inconsistency (premium plan but enterprise pricing)
        {"id": 6,  "name": "Frank Miller",      "email": "frank.m@domain.org",         "age": 67,   "city": "Philadelphia",  "signup_date": "2024-02-14", "plan": "premium",    "monthly_amount": 99.99, "status": "active"},
        # Row 7: CLEAN
        {"id": 7,  "name": "Grace Wilson",      "email": "grace.wilson@mail.com",      "age": 23,   "city": "San Antonio",   "signup_date": "2024-05-01", "plan": "free",       "monthly_amount": 0.00,  "status": "active"},
        # Row 8: ERROR - missing value (empty city)
        {"id": 8,  "name": "Henry Taylor",      "email": "henry.t@inbox.com",          "age": 41,   "city": "",              "signup_date": "2024-03-22", "plan": "basic",      "monthly_amount": 9.99,  "status": "cancelled"},
        # Row 9: ERROR - type error (age is a string)
        {"id": 9,  "name": "Ivy Anderson",      "email": "ivy.anderson@webmail.com",   "age": "thirty-six", "city": "Dallas", "signup_date": "2024-06-11", "plan": "enterprise", "monthly_amount": 99.99, "status": "active"},
        # Row 10: ERROR - duplicate of row 1 (same person, different ID)
        {"id": 10, "name": "Alice Johnson",     "email": "alice.johnson@gmail.com",    "age": 34,   "city": "New York",      "signup_date": "2024-01-15", "plan": "premium",    "monthly_amount": 29.99, "status": "active"},
        # Row 11: CLEAN
        {"id": 11, "name": "Karen Martinez",    "email": "karen.m@provider.com",       "age": 55,   "city": "Austin",        "signup_date": "2024-01-07", "plan": "basic",      "monthly_amount": 9.99,  "status": "active"},
        # Row 12: ERROR - invalid status value
        {"id": 12, "name": "Leo Garcia",        "email": "leo.garcia@fastmail.com",    "age": 19,   "city": "Jacksonville",  "signup_date": "2024-07-03", "plan": "free",       "monthly_amount": 0.00,  "status": "actve"},
        # Row 13: ERROR - impossible age (outlier, too high)
        {"id": 13, "name": "Mia Robinson",      "email": "mia.r@outlook.com",          "age": 250,  "city": "Fort Worth",    "signup_date": "2024-02-28", "plan": "premium",    "monthly_amount": 29.99, "status": "suspended"},
        # Row 14: CLEAN
        {"id": 14, "name": "Noah Clark",        "email": "noah.clark@gmail.com",       "age": 33,   "city": "Columbus",      "signup_date": "2024-05-19", "plan": "enterprise", "monthly_amount": 99.99, "status": "active"},
        # Row 15: ERROR - missing email (null value)
        {"id": 15, "name": "Olivia Lewis",      "email": "",                           "age": 27,   "city": "Charlotte",     "signup_date": "2024-08-10", "plan": "basic",      "monthly_amount": 9.99,  "status": "active"},
    ]


def get_ground_truth_errors() -> List[Dict[str, Any]]:
    """
    Return the ground truth list of all planted errors.
    
    Each error entry contains:
        - row_id: 1-indexed row number
        - column: Column name where the error is
        - error_type: Category of the error
        - current_value: The erroneous value in the dirty dataset
        - corrected_value: The correct value that should be there
        - description: Human-readable explanation of the error
    """
    return [
        {
            "row_id": 2,
            "column": "email",
            "error_type": "invalid_format",
            "current_value": "bob.smith@yahoo",
            "corrected_value": "bob.smith@yahoo.com",
            "description": "Email is missing a top-level domain (TLD). Should be 'bob.smith@yahoo.com'.",
        },
        {
            "row_id": 3,
            "column": "signup_date",
            "error_type": "invalid_format",
            "current_value": "2024-02-30",
            "corrected_value": "2024-02-28",
            "description": "February 30th does not exist. The latest valid date in Feb 2024 is 2024-02-28 (leap year) or 2024-02-29.",
        },
        {
            "row_id": 5,
            "column": "age",
            "error_type": "outlier",
            "current_value": -3,
            "corrected_value": 31,
            "description": "Age is negative (-3), which is impossible. Correct value is 31.",
        },
        {
            "row_id": 6,
            "column": "monthly_amount",
            "error_type": "inconsistency",
            "current_value": 99.99,
            "corrected_value": 29.99,
            "description": "Premium plan costs $29.99/month, not $99.99. Amount is inconsistent with the plan.",
        },
        {
            "row_id": 8,
            "column": "city",
            "error_type": "missing_value",
            "current_value": "",
            "corrected_value": "San Diego",
            "description": "City field is empty. Should be 'San Diego'.",
        },
        {
            "row_id": 9,
            "column": "age",
            "error_type": "type_error",
            "current_value": "thirty-six",
            "corrected_value": 36,
            "description": "Age is stored as a word string 'thirty-six' instead of the integer 36.",
        },
        {
            "row_id": 10,
            "column": "name",
            "error_type": "duplicate",
            "current_value": "Alice Johnson",
            "corrected_value": "Jack Thomas",
            "description": "Row 10 is an exact duplicate of Row 1 (same name, email, age, city, date, plan). Should be a unique customer 'Jack Thomas'.",
        },
        {
            "row_id": 12,
            "column": "status",
            "error_type": "invalid_format",
            "current_value": "actve",
            "corrected_value": "active",
            "description": "Status 'actve' is a typo. Should be 'active'.",
        },
        {
            "row_id": 13,
            "column": "age",
            "error_type": "outlier",
            "current_value": 250,
            "corrected_value": 48,
            "description": "Age 250 is impossibly high. Correct value is 48.",
        },
        {
            "row_id": 15,
            "column": "email",
            "error_type": "missing_value",
            "current_value": "",
            "corrected_value": "olivia.l@yahoo.com",
            "description": "Email field is empty. Should be 'olivia.l@yahoo.com'.",
        },
    ]


def get_error_row_ids() -> List[int]:
    """Return just the row IDs that contain errors."""
    return sorted(set(e["row_id"] for e in get_ground_truth_errors()))


def format_dataset_as_table(dataset: List[Dict[str, Any]]) -> str:
    """
    Format the dataset as a readable text table for the agent.
    
    Returns a plain-text table with aligned columns.
    """
    if not dataset:
        return "(empty dataset)"

    headers = COLUMNS
    
    # Calculate column widths
    col_widths = {}
    for col in headers:
        values = [str(row.get(col, "")) for row in dataset]
        col_widths[col] = max(len(col), max(len(v) for v in values)) + 2

    # Build header row
    header_line = "| " + " | ".join(col.ljust(col_widths[col]) for col in headers) + " |"
    separator = "|-" + "-|-".join("-" * col_widths[col] for col in headers) + "-|"

    # Build data rows
    data_lines = []
    for row in dataset:
        line = "| " + " | ".join(str(row.get(col, "")).ljust(col_widths[col]) for col in headers) + " |"
        data_lines.append(line)

    return "\n".join([header_line, separator] + data_lines)


def get_dataset_summary() -> str:
    """Return a summary of the dataset structure and validation rules."""
    return """Dataset: Customer Subscription Records
Columns: id, name, email, age, city, signup_date, plan, monthly_amount, status
Total rows: 15

Validation Rules:
- id: Sequential integers, must be unique
- name: Non-empty string in 'First Last' format
- email: Valid email format (must contain @ and a valid TLD like .com, .org, .net, etc.)
- age: Integer between 18 and 100
- city: Non-empty string, valid US city name
- signup_date: Valid date in YYYY-MM-DD format (no impossible dates like Feb 30)
- plan: Must be one of: free, basic, premium, enterprise
- monthly_amount: Must match plan pricing (free=$0.00, basic=$9.99, premium=$29.99, enterprise=$99.99)
- status: Must be one of: active, inactive, suspended, cancelled
- No duplicate rows (same name+email = duplicate)"""
