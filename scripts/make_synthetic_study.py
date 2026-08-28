"""Generate a synthetic SDTM-shaped study for development and testing.

Entirely fabricated: no real subject, site, or investigator appears here. The
free-text columns deliberately contain planted identifiers of the kind that
turn up in real verbatim fields -- hospital names, physician names, phone
numbers, dates written out longhand -- so the screening pass has something to
find and the review queue can be inspected.

Usage:  python scripts/make_synthetic_study.py out/quarantine/study_demo
"""

from __future__ import annotations

import random
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

SEED = 20260827
N_SUBJECTS = 120

SITES = ["001", "002", "003", "004", "005", "017"]
# Site 017 enrols a single subject on purpose: a site that small is itself
# identifying, and the risk report should surface it.
SITE_WEIGHTS = [30, 28, 24, 20, 16, 1]

SEXES = ["M", "F"]
RACES = [
    "WHITE",
    "BLACK OR AFRICAN AMERICAN",
    "ASIAN",
    "AMERICAN INDIAN OR ALASKA NATIVE",
    "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER",
]
RACE_WEIGHTS = [58, 20, 16, 4, 2]
ETHNIC = ["NOT HISPANIC OR LATINO", "HISPANIC OR LATINO"]
# Real drug names, deliberately. A placeholder like "Drug A" makes it
# impossible to see from the output whether the pipeline masks drug names --
# and the answer matters: it does not, because a treatment arm is the exposure
# under study, identical for everyone in it, and identifies no one.
ARMS = [
    "Placebo",
    "Pembrolizumab 200 mg Q3W",
    "Pembrolizumab 400 mg Q6W",
]

# Concomitant medications: real names again, and the verbatim field is where
# investigators type free text, so it is also where identifiers turn up.
CM_MEDS = [
    ("Lisinopril", "LISINOPRIL", "Hypertension", 10, "mg", "ORAL"),
    ("Metformin HCl 500mg", "METFORMIN HYDROCHLORIDE", "Type 2 diabetes", 500, "mg", "ORAL"),
    ("Atorvastatin", "ATORVASTATIN", "Hyperlipidaemia", 20, "mg", "ORAL"),
    ("Levothyroxine sodium", "LEVOTHYROXINE SODIUM", "Hypothyroidism", 75, "ug", "ORAL"),
    ("Amlodipine besylate", "AMLODIPINE BESILATE", "Hypertension", 5, "mg", "ORAL"),
    ("Omeprazole", "OMEPRAZOLE", "GERD", 20, "mg", "ORAL"),
    ("Paracetamol prn", "PARACETAMOL", "Headache", 500, "mg", "ORAL"),
    ("Ondansetron", "ONDANSETRON", "Nausea", 8, "mg", "INTRAVENOUS"),
    ("Warfarin sodium", "WARFARIN SODIUM", "Atrial fibrillation", 5, "mg", "ORAL"),
    ("Adalimumab", "ADALIMUMAB", "Crohn's disease", 40, "mg", "SUBCUTANEOUS"),
]

# Verbatim conmed entries carrying identifiers -- the realistic failure mode.
CM_MEDS_WITH_PHI = [
    ("Insulin glargine, started by Dr. Halvorsen at Riverside Clinic",
     "INSULIN GLARGINE", "Type 2 diabetes", 20, "U", "SUBCUTANEOUS"),
    ("Amoxicillin prescribed 04/12/2026 by GP, see fax 617-555-0198",
     "AMOXICILLIN", "Infection", 500, "mg", "ORAL"),
]

AE_TERMS = [
    ("Headache", "Headache", "Nervous system disorders"),
    ("Nausea", "Nausea", "Gastrointestinal disorders"),
    ("mild nausea after dosing", "Nausea", "Gastrointestinal disorders"),
    ("Fatigue", "Fatigue", "General disorders"),
    ("Dizziness", "Dizziness", "Nervous system disorders"),
    ("Upper resp infection", "Upper respiratory tract infection", "Infections"),
    ("Rash on forearm", "Rash", "Skin disorders"),
    ("Diarrhoea", "Diarrhoea", "Gastrointestinal disorders"),
    ("Insomnia", "Insomnia", "Psychiatric disorders"),
    ("Back pain", "Back pain", "Musculoskeletal disorders"),
    ("Pyrexia", "Pyrexia", "General disorders"),
    ("Elevated ALT", "Alanine aminotransferase increased", "Investigations"),
    # rare terms -- retained by decision, but quasi-identifying
    ("Guillain-Barre syndrome", "Guillain-Barre syndrome", "Nervous system disorders"),
    ("Stevens-Johnson syndrome", "Stevens-Johnson syndrome", "Skin disorders"),
]

# Verbatim strings with planted identifiers. These are what the screen must
# catch, and the surrounding clinical text is what must survive untouched.
AE_TERMS_WITH_PHI = [
    ("Fell at St Mary's Hospital, seen by Dr. Okafor", "Fall", "Injury"),
    ("Admitted to Mercy General Medical Center on 03/14/2026", "Hospitalisation", "General disorders"),
    ("Subject called site nurse at (617) 555-0142 reporting chest pain", "Chest pain", "Cardiac disorders"),
    ("Rash worsened; photo emailed to j.almeida@sitemail.example", "Rash", "Skin disorders"),
    ("Seen in ER at Northside Clinic, discharged same day", "Emergency room visit", "General disorders"),
]

# Verbatim naming the study treatment. Relabelling ARM does nothing about
# these, and free text is where a language model memorises -- so this is the
# blinding leak that actually matters for a training corpus.
AE_TERMS_UNBLINDING = [
    ("Rash 3 days after pembrolizumab infusion", "Rash", "Skin disorders"),
    ("Fatigue, subject asked to skip next Pembrolizumab 200 mg Q3W dose",
     "Fatigue", "General disorders"),
]

MH_TERMS = [
    ("Hypertension", "Hypertension"),
    ("Type 2 diabetes", "Type 2 diabetes mellitus"),
    ("Asthma", "Asthma"),
    ("Osteoarthritis", "Osteoarthritis"),
    ("Hypothyroidism", "Hypothyroidism"),
    ("Depression", "Depression"),
    ("GERD", "Gastrooesophageal reflux disease"),
    ("Crohn disease", "Crohn's disease"),
    ("Migraine", "Migraine"),
    ("Atrial fibrillation", "Atrial fibrillation"),
]

LB_TESTS = [
    ("ALT", "Alanine Aminotransferase", "U/L", 10, 55),
    ("AST", "Aspartate Aminotransferase", "U/L", 10, 48),
    ("CREAT", "Creatinine", "umol/L", 55, 110),
    ("HGB", "Hemoglobin", "g/dL", 11.5, 16.5),
    ("PLAT", "Platelets", "10^9/L", 150, 400),
]


def iso(d: date) -> str:
    return d.isoformat()


def main(outdir: str) -> None:
    rng = random.Random(SEED)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    study = "TIG-2026-001"
    dm_rows, ae_rows, mh_rows, lb_rows, vs_rows, cm_rows, ex_rows = [], [], [], [], [], [], []

    for i in range(1, N_SUBJECTS + 1):
        site = rng.choices(SITES, weights=SITE_WEIGHTS, k=1)[0]
        subjid = f"{site}-{i:04d}"
        usubjid = f"{study}-US-{subjid}"

        # enrolment spread over ~14 months
        enrol = date(2025, 1, 6) + timedelta(days=rng.randrange(0, 430))
        # ages skewed older, with a few over 89 so the 90+ cap is exercised
        age = min(94, max(19, int(rng.gauss(58, 15))))
        birth = date(enrol.year - age, rng.randrange(1, 13), rng.randrange(1, 28))
        sex = rng.choice(SEXES)
        race = rng.choices(RACES, weights=RACE_WEIGHTS, k=1)[0]
        arm = rng.choice(ARMS)
        end = enrol + timedelta(days=rng.randrange(60, 380))

        dm_rows.append(
            {
                "STUDYID": study,
                "DOMAIN": "DM",
                "USUBJID": usubjid,
                "SUBJID": subjid,
                "SITEID": site,
                "INVNAM": rng.choice(
                    ["Dr. A. Whitfield", "Dr. M. Sørensen", "Dr. P. Raghunathan"]
                ),
                "BRTHDTC": iso(birth),
                "AGE": age,
                "AGEU": "YEARS",
                "SEX": sex,
                "RACE": race,
                "ETHNIC": rng.choices(ETHNIC, weights=[80, 20], k=1)[0],
                "COUNTRY": "USA",
                "SITEZIP": rng.choice(
                    ["02115", "10029", "60637", "94143", "03601", "77030"]
                ),
                "ARM": arm,
                "RFSTDTC": iso(enrol),
                "RFENDTC": iso(end),
                "SUBJPHONE": f"({rng.randrange(200,990)}) 555-{rng.randrange(1000,9999)}",
            }
        )

        # --- adverse events ------------------------------------------
        for seq in range(1, rng.randrange(1, 6)):
            r = rng.random()
            if r < 0.06:
                verbatim, decod, soc = rng.choice(AE_TERMS_WITH_PHI)
            elif r < 0.10:
                verbatim, decod, soc = rng.choice(AE_TERMS_UNBLINDING)
            else:
                verbatim, decod, soc = rng.choice(AE_TERMS)
            start = enrol + timedelta(days=rng.randrange(1, 300))
            ongoing = rng.random() < 0.15
            aeend = None if ongoing else start + timedelta(days=rng.randrange(1, 40))
            ae_rows.append(
                {
                    "STUDYID": study,
                    "DOMAIN": "AE",
                    "USUBJID": usubjid,
                    "AESEQ": seq,
                    "AETERM": verbatim,
                    "AEDECOD": decod,
                    "AEBODSYS": soc,
                    "AESEV": rng.choices(
                        ["MILD", "MODERATE", "SEVERE"], weights=[60, 30, 10], k=1
                    )[0],
                    "AESER": "Y" if rng.random() < 0.08 else "N",
                    "AEREL": rng.choices(
                        ["NOT RELATED", "POSSIBLY RELATED", "RELATED"],
                        weights=[50, 35, 15],
                        k=1,
                    )[0],
                    "AEOUT": rng.choice(
                        ["RECOVERED/RESOLVED", "RECOVERING/RESOLVING", "NOT RECOVERED"]
                    ),
                    "AEACN": rng.choice(
                        ["DOSE NOT CHANGED", "DRUG INTERRUPTED", "DRUG WITHDRAWN"]
                    ),
                    "AESTDTC": iso(start),
                    "AEENDTC": iso(aeend) if aeend else None,
                    "AEENRF": "ONGOING" if ongoing else None,
                }
            )

        # --- medical history: partial dates on purpose ----------------
        for seq in range(1, rng.randrange(1, 5)):
            verbatim, decod = rng.choice(MH_TERMS)
            years_ago = rng.randrange(1, 25)
            r = rng.random()
            if r < 0.45:  # year only -- the common real-world case
                mhst = str(enrol.year - years_ago)
            elif r < 0.70:  # year-month
                mhst = f"{enrol.year - years_ago}-{rng.randrange(1, 13):02d}"
            else:  # complete
                mhst = iso(
                    date(
                        enrol.year - years_ago,
                        rng.randrange(1, 13),
                        rng.randrange(1, 28),
                    )
                )
            mh_rows.append(
                {
                    "STUDYID": study,
                    "DOMAIN": "MH",
                    "USUBJID": usubjid,
                    "MHSEQ": seq,
                    "MHTERM": verbatim,
                    "MHDECOD": decod,
                    "MHBODSYS": "Medical history",
                    "MHSTDTC": mhst,
                    "MHONGO": rng.choice(["Y", "N"]),
                }
            )

        # --- concomitant medications ---------------------------------
        for seq in range(1, rng.randrange(1, 5)):
            if rng.random() < 0.05:
                verbatim, decod, indc, dose, unit, route = rng.choice(CM_MEDS_WITH_PHI)
            else:
                verbatim, decod, indc, dose, unit, route = rng.choice(CM_MEDS)
            start = enrol - timedelta(days=rng.randrange(0, 900))
            cm_rows.append(
                {
                    "STUDYID": study,
                    "DOMAIN": "CM",
                    "USUBJID": usubjid,
                    "CMSEQ": seq,
                    "CMTRT": verbatim,
                    "CMDECOD": decod,
                    "CMINDC": indc,
                    "CMDOSE": dose,
                    "CMDOSU": unit,
                    "CMROUTE": route,
                    "CMSTDTC": iso(start),
                }
            )

        # --- exposure: dose and regimen give the arm away ------------
        dose, freq = {
            "Placebo": (0, "Q3W"),
            "Pembrolizumab 200 mg Q3W": (200, "Q3W"),
            "Pembrolizumab 400 mg Q6W": (400, "Q6W"),
        }[arm]
        for seq in range(1, 4):
            ex_rows.append(
                {
                    "STUDYID": study,
                    "DOMAIN": "EX",
                    "USUBJID": usubjid,
                    "EXSEQ": seq,
                    "EXTRT": arm.split(" ")[0],
                    "EXDOSE": dose,
                    "EXDOSU": "mg",
                    "EXDOSFRQ": freq,
                    "EXROUTE": "INTRAVENOUS",
                    "EXSTDTC": iso(enrol + timedelta(days=21 * (seq - 1))),
                }
            )

        # --- labs and vitals: the analytic payload -------------------
        for visit_day in (1, 29, 57, 85):
            vdate = enrol + timedelta(days=visit_day - 1)
            for code, name, unit, lo, hi in LB_TESTS:
                lb_rows.append(
                    {
                        "STUDYID": study,
                        "DOMAIN": "LB",
                        "USUBJID": usubjid,
                        "LBSEQ": len(lb_rows) + 1,
                        "LBTESTCD": code,
                        "LBTEST": name,
                        "LBORRES": round(rng.uniform(lo * 0.7, hi * 1.3), 2),
                        "LBORRESU": unit,
                        "VISITNUM": visit_day,
                        "LBDTC": iso(vdate),
                    }
                )
            vs_rows.append(
                {
                    "STUDYID": study,
                    "DOMAIN": "VS",
                    "USUBJID": usubjid,
                    "VSSEQ": len(vs_rows) + 1,
                    "VSTESTCD": "SYSBP",
                    "VSTEST": "Systolic Blood Pressure",
                    "VSORRES": int(rng.gauss(128, 14)),
                    "VSORRESU": "mmHg",
                    "VISITNUM": visit_day,
                    "VSDTC": iso(vdate),
                }
            )

    for name, rows in (
        ("dm", dm_rows),
        ("ae", ae_rows),
        ("mh", mh_rows),
        ("lb", lb_rows),
        ("vs", vs_rows),
        ("cm", cm_rows),
        ("ex", ex_rows),
    ):
        frame = pd.DataFrame(rows)
        frame.to_csv(out / f"{name}.csv", index=False)
        print(f"{name.upper():<4} {len(frame):>6} rows  {len(frame.columns):>3} cols")

    print(f"\nwrote synthetic study to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out/quarantine/study_demo")
