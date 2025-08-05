import os
import re
import pdfplumber
import logging
import json
from typing import Dict, List
from .utils import normalize_bureau_name, BUREAUS
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv

logging.getLogger("pdfplumber.page").setLevel(logging.ERROR)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_clean_name(full_name: str) -> str:
    parts = full_name.strip().split()
    seen = set()
    unique_parts = []
    for part in parts:
        if part.lower() not in seen:
            unique_parts.append(part)
            seen.add(part.lower())
    return " ".join(unique_parts)

def normalize_name_order(name: str) -> str:
    parts = name.strip().split()
    if len(parts) == 2:
        return f"{parts[1]} {parts[0]}"
    return name

def extract_bureau_info_column_refined(pdf_path: str, client_info: dict = None, use_ai: bool = False) -> Dict[str, Dict[str, str]]:
    bureaus = BUREAUS
    data = {b: {"name": "", "dob": "", "current_address": ""} for b in bureaus}
    discrepancies = []

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        words = page.extract_words()
        raw_text = page.extract_text()

    columns = {b: [] for b in bureaus}
    for w in words:
        x0 = float(w["x0"])
        if x0 < 200:
            columns[normalize_bureau_name("TransUnion")].append(w)
        elif x0 < 400:
            columns[normalize_bureau_name("Experian")].append(w)
        else:
            columns[normalize_bureau_name("Equifax")].append(w)

    noise_words = {
        "account", "progress", "plan", "reactivate", "score", "alert", "change",
        "your", "personal", "information", "identity", "elements", "name",
        "employer", "lowes", "consumer", "statement"
    }

    def group_words_by_line(words: List[Dict]) -> List[str]:
        lines = []
        current_line = []
        last_top = None
        for w in sorted(words, key=lambda x: (x["top"], x["x0"])):
            if last_top is None or abs(w["top"] - last_top) < 3:
                current_line.append(w["text"])
                last_top = w["top"]
            else:
                lines.append(" ".join(current_line))
                current_line = [w["text"]]
                last_top = w["top"]
        if current_line:
            lines.append(" ".join(current_line))
        return lines

    def extract_full_name(lines: List[str]) -> str:
        for line in lines:
            line_upper = line.upper()
            words = line.strip().split()
            if (
                2 <= len(words) <= 5 and
                all(len(w) > 1 for w in words) and
                not any(noise in line_upper.lower() for noise in noise_words)
            ):
                if re.fullmatch(r"[A-Z\s\.']{6,}", line_upper):
                    return extract_clean_name(line.title())
        return ""

    def extract_dob(lines: List[str]) -> str:
        for line in lines:
            match = re.search(r"\b(19|20)\d{2}\b", line)
            if match and match.group(0) != "2025":
                return match.group(0)
        return ""

    def extract_address(lines: List[str]) -> str:
        for i in range(len(lines) - 1):
            combined = f"{lines[i]} {lines[i+1]}"
            if re.search(r"\d{3,5} .+ [A-Z]{2} \d{5}", combined.upper()):
                return combined.title()
        return ""

    for bureau in bureaus:
        col_lines = group_words_by_line(columns[bureau])
        data[bureau]["name"] = extract_full_name(col_lines)
        data[bureau]["dob"] = extract_dob(col_lines)
        data[bureau]["current_address"] = extract_address(col_lines)

    for field in ["name", "dob", "current_address"]:
        field_values = [data[b][field] for b in bureaus if data[b][field]]
        if len(field_values) >= 2:
            most_common = Counter(field_values).most_common(1)[0][0]
            for b in bureaus:
                if not data[b][field] or (
                    len(data[b][field].split()) < 2 and
                    data[b][field].lower() not in most_common.lower()
                ):
                    data[b][field] = most_common

    # Normalize name order only once
    for b in bureaus:
        data[b]["name"] = normalize_name_order(data[b]["name"])

    for field in ["name", "dob", "current_address"]:
        values = {b: data[b][field].strip().lower() for b in bureaus if data[b][field]}
        if len(set(values.values())) > 1:
            discrepancies.append(
                f"⚠️ Mismatch in {field} across bureaus:\n" +
                "\n".join([f"  - {b}: {data[b][field]}" for b in bureaus])
            )

    # Compare to client-provided info
    if client_info:
        if "legal_name" in client_info:
            extracted = data["Experian"]["name"].lower().strip()
            legal = client_info["legal_name"].lower().strip()
            if set(extracted.split()) != set(legal.split()):
                discrepancies.append(f"⚠️ Name mismatch with ID: extracted '{extracted}' vs client '{legal}'")

        if "legal_address" in client_info:
            extracted = data["Experian"]["current_address"].lower().strip()
            legal = client_info["legal_address"].lower().strip()
            if extracted != legal:
                discrepancies.append(f"⚠️ Address mismatch with ID: extracted '{extracted}' vs client '{legal}'")

    if use_ai:
        try:
            print("[🤖] Running GPT validation for personal info...")
            prompt = f"""
You are a credit repair AI assistant.
You received the first page of a SmartCredit report. Extract the following:

- Full name (most consistent one)
- Year of birth (YYYY)
- Current full address

Output JSON in this format:
{{
  "name": "...",
  "dob": "...",
  "current_address": "..."
}}

Here is the text:
===
{raw_text}
===
"""
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content.strip()
            ai_data = json.loads(content)
            for b in bureaus:
                data[b].update(ai_data)
        except Exception as e:
            print(f"[⚠️] AI info extraction failed: {str(e)}")

    return {
        "data": data["Experian"],
        "discrepancies": discrepancies,
        "raw_all_bureaus": data
    }
