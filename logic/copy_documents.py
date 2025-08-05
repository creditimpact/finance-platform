import os
import shutil
from pathlib import Path

def copy_required_documents(client_info, client_folder: Path, proofs_files: dict, is_identity_theft: bool):
    bureaus = ["Experian", "Equifax", "TransUnion"]

    for bureau in bureaus:
        bureau_folder = client_folder / bureau
        bureau_folder.mkdir(parents=True, exist_ok=True)

        # Copy dispute letter
        dispute_letter = client_folder / f"Dispute Letter - {bureau}.pdf"
        if dispute_letter.exists():
            shutil.copy(dispute_letter, bureau_folder / dispute_letter.name)

    # Copy goodwill letters (once, not per bureau)
    goodwill_folder = client_folder / "goodwill_letters"
    if goodwill_folder.exists():
        for f in goodwill_folder.glob("*.pdf"):
            shutil.copy(f, client_folder / f.name)

    if is_identity_theft:
        # Copy FTC FCRA document
        static_doc = Path("templates") / "FTC_FCRA_605b.pdf"
        if static_doc.exists():
            shutil.copy(static_doc, client_folder / "Your Rights - FCRA.pdf")
        else:
            print("[⚠️] FCRA rights file not found!")

    # Copy user-uploaded SmartCredit report
    if "smartcredit_report" in proofs_files:
        shutil.copy(proofs_files["smartcredit_report"], client_folder / "SmartCredit_Report.pdf")

    print(f"[📂] Documents organized in: {client_folder}")
