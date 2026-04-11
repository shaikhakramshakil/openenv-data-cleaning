# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Synthetic dataset with planted data quality errors.

Includes a dynamic generator that can create datasets of any size with
configurable error rates and advanced error types.
"""

import random
from typing import Any, Dict, List, Tuple
from datetime import datetime, timedelta

# ─── Schema Definitions ────────────────────────────────────────────────

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

PLAN_PRICING = {
    "free": 0.00,
    "basic": 9.99,
    "premium": 29.99,
    "enterprise": 99.99,
}

# ─── Seed Data for Generation ──────────────────────────────────────────

FIRST_NAMES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Ivy", "Jack", "Karen", "Leo", "Mia", "Noah", "Olivia", "Paul", "Quinn", "Rose", "Sam", "Tara"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"]
CITIES = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville", "Fort Worth", "Columbus", "Charlotte", "San Francisco", "Indianapolis", "Seattle", "Denver", "Washington"]
PLANS = list(PLAN_PRICING.keys())
STATUSES = ["active", "inactive", "suspended", "cancelled"]

# ─── Generator Class ──────────────────────────────────────────────────

class DatasetGenerator:
    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)

    def generate_row(self, row_id: int) -> Dict[str, Any]:
        first = self.random.choice(FIRST_NAMES)
        last = self.random.choice(LAST_NAMES)
        name = f"{first} {last}"
        email = f"{first.lower()}.{last.lower()}@{self.random.choice(['gmail.com', 'yahoo.com', 'outlook.com', 'company.co', 'mail.com'])}"
        age = self.random.randint(18, 75)
        city = self.random.choice(CITIES)
        
        # Signup date within the last 2 years
        days_ago = self.random.randint(0, 730)
        signup_date = (datetime(2024, 8, 1) - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        
        plan = self.random.choice(PLANS)
        amount = PLAN_PRICING[plan]
        status = self.random.choice(STATUSES)
        
        return {
            "id": row_id,
            "name": name,
            "email": email,
            "age": age,
            "city": city,
            "signup_date": signup_date,
            "plan": plan,
            "monthly_amount": amount,
            "status": status,
        }

    def create_dataset(self, n_rows: int = 50, error_rate: float = 0.2) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Generates a clean and a dirty dataset.
        Returns: (dirty_dataset, ground_truth_errors)
        """
        dataset = [self.generate_row(i + 1) for i in range(n_rows)]
        dirty_dataset = [row.copy() for row in dataset]
        ground_truth_errors = []

        # Target number of errors
        n_errors = int(n_rows * error_rate * 2) # Roughly 2 errors per "dirty" row selection
        
        error_types = ["missing_value", "invalid_format", "outlier", "duplicate", "type_error", "inconsistency"]
        
        affected_rows = self.random.sample(range(n_rows), int(n_rows * error_rate))
        
        for row_idx in affected_rows:
            row_id = row_idx + 1
            error_type = self.random.choice(error_types)
            col = self.random.choice(COLUMNS[1:]) # Skip ID
            
            error_info = {
                "row_id": row_id,
                "column": col,
                "error_type": error_type,
                "current_value": dirty_dataset[row_idx][col],
                "corrected_value": dataset[row_idx][col],
                "description": ""
            }

            if error_type == "missing_value":
                dirty_dataset[row_idx][col] = "" if self.random.random() > 0.5 else None
                error_info["current_value"] = dirty_dataset[row_idx][col]
                error_info["description"] = f"Value in '{col}' is missing."
            
            elif error_type == "invalid_format":
                if col == "email":
                    dirty_dataset[row_idx][col] = dirty_dataset[row_idx][col].split("@")[0] + "@" + self.random.choice(["gmail", "yahoo", "outlook"])
                    error_info["description"] = "Email is missing a top-level domain (TLD)."
                elif col == "signup_date":
                    dirty_dataset[row_idx][col] = dirty_dataset[row_idx][col][:8] + "32" # Invalid day
                    error_info["description"] = "Date has an impossible day (32)."
                elif col == "status":
                    dirty_dataset[row_idx][col] = dirty_dataset[row_idx][col][:-1] # Typo
                    error_info["description"] = f"Typo in status value: '{dirty_dataset[row_idx][col]}'."
                else:
                    dirty_dataset[row_idx][col] = "???"
                    error_info["description"] = f"Invalid format in '{col}'."
                error_info["current_value"] = dirty_dataset[row_idx][col]

            elif error_type == "outlier":
                if col == "age":
                    val = self.random.choice([-5, 150, 250, 0])
                    dirty_dataset[row_idx][col] = val
                    error_info["description"] = f"Age {val} is outside the valid range (18-100)."
                elif col == "monthly_amount":
                    val = 9999.99
                    dirty_dataset[row_idx][col] = val
                    error_info["description"] = f"Amount ${val} is an extreme outlier for this plan."
                else:
                    dirty_dataset[row_idx][col] = 999
                    error_info["description"] = "Value is an outlier."
                error_info["current_value"] = dirty_dataset[row_idx][col]

            elif error_type == "duplicate":
                # Duplicate another clean row but keep this unique ID
                source_idx = (row_idx + 1) % n_rows
                for c in COLUMNS[1:]: # Copy everything except ID
                    dirty_dataset[row_idx][c] = dataset[source_idx][c]
                error_info["column"] = "ALL"
                error_info["description"] = f"Row {row_id} is a duplicate of Row {source_idx + 1}."
                error_info["current_value"] = "Duplicate Row"
                error_info["corrected_value"] = "Unique Data"

            elif error_type == "type_error":
                if col == "age":
                    dirty_dataset[row_idx][col] = "twenty"
                elif col == "monthly_amount":
                    dirty_dataset[row_idx][col] = "unknown"
                else:
                    dirty_dataset[row_idx][col] = True
                error_info["current_value"] = dirty_dataset[row_idx][col]
                error_info["description"] = f"Wrong data type in '{col}'."

            elif error_type == "inconsistency":
                if col == "monthly_amount":
                    # Mismatch plan and amount
                    wrong_plan = self.random.choice([p for p in PLANS if p != dirty_dataset[row_idx]["plan"]])
                    dirty_dataset[row_idx]["monthly_amount"] = PLAN_PRICING[wrong_plan]
                    error_info["description"] = f"Price ${dirty_dataset[row_idx]['monthly_amount']} does not match plan '{dirty_dataset[row_idx]['plan']}'."
                else:
                    dirty_dataset[row_idx][col] = "inconsistent"
                    error_info["description"] = "Logic inconsistency detected."
                error_info["current_value"] = dirty_dataset[row_idx][col]

            ground_truth_errors.append(error_info)

        return dirty_dataset, ground_truth_errors

# ─── Global State for "Standard" Environment ─────────────────────────

_gen = DatasetGenerator(seed=42)
_dirty_data, _ground_truth = _gen.create_dataset(n_rows=30, error_rate=0.4) # Doubled size and complexity from previous

def get_clean_dataset() -> List[Dict[str, Any]]:
    # Just for reference, we don't store the full clean set usually
    return [] 

def get_dirty_dataset() -> List[Dict[str, Any]]:
    return _dirty_data

def get_ground_truth_errors() -> List[Dict[str, Any]]:
    return _ground_truth

def get_error_row_ids() -> List[int]:
    return sorted(set(e["row_id"] for e in _ground_truth))

def format_dataset_as_table(dataset: List[Dict[str, Any]]) -> str:
    if not dataset:
        return "(empty dataset)"
    headers = COLUMNS
    col_widths = {col: max(len(col), max((len(str(row.get(col, ""))) for row in dataset), default=0)) + 2 for col in headers}
    header_line = "| " + " | ".join(col.ljust(col_widths[col]) for col in headers) + " |"
    separator = "|-" + "-|-".join("-" * col_widths[col] for col in headers) + "-|"
    data_lines = ["| " + " | ".join(str(row.get(col, "")).ljust(col_widths[col]) for col in headers) + " |" for row in dataset]
    return "\n".join([header_line, separator] + data_lines)

def get_dataset_summary() -> str:
    return f"""Dataset: Customer Subscription Records (V2 - High Complexity)
Columns: {', '.join(COLUMNS)}
Total rows: {len(_dirty_data)}

Validation Rules:
- id: Sequential integers, must be unique
- name: Non-empty string in 'First Last' format
- email: Valid email format (must contain @ and a valid TLD like .com, .org, .net, etc.)
- age: Integer between 18 and 100
- city: Non-empty string, valid US city name
- signup_date: Valid date in YYYY-MM-DD format
- plan: Must be one of: {', '.join(PLANS)}
- monthly_amount: Must match plan pricing (free=$0.00, basic=$9.99, premium=$29.99, enterprise=$99.99)
- status: Must be one of: {', '.join(STATUSES)}
- No duplicate rows (same identity across different IDs)"""

def get_validation_rules() -> Dict[str, Any]:
    return {
        "columns": COLUMN_DESCRIPTIONS,
        "pricing": PLAN_PRICING,
        "plans": PLANS,
        "statuses": STATUSES,
        "cities": CITIES
    }
