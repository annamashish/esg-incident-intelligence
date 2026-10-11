"""Generate a clearly-labelled SYNTHETIC extension of the bundled incident dataset.

The bundled CSV only has a handful of rows, which is not enough to train or
validate a classifier. This script writes data/incidents_extended.csv with
template-based narratives (source="synthetic") for renewable-energy sites.

Because the rows come from templates, metrics measured on them are optimistic.
The README reports this honestly and also evaluates on the real bundled rows.

Run:  python scripts/generate_data.py
"""
import random
from pathlib import Path

import pandas as pd

random.seed(42)
OUT = Path(__file__).resolve().parents[1] / "data" / "incidents_extended.csv"

SITES = ["Solar Park A", "Solar Park B", "Wind Farm North", "Wind Farm Ridge",
         "Hydro Plant 1", "Substation 220kV", "BESS Yard"]
ASSETS = ["inverter station", "wind turbine", "cable trench", "transformer yard",
          "module cleaning area", "gearbox bay", "battery container", "control room",
          "access road", "diesel generator", "substation bay"]

# (category, severity) -> list of templates. {a}=asset, {s}=site
T = {
    ("safety", "low"): [
        "Minor first aid case, technician scraped hand while opening panel at {a}, {s}",
        "Housekeeping lapse noticed near {a}, loose tools left on the floor",
        "Worker reported a minor trip over a cable cover at {a}, no injury",
        "Safety signage missing at {a}, replaced the same day",
        "Toolbox talk not recorded for morning shift at {a}",
    ],
    ("safety", "medium"): [
        "PPE non-compliance observed, helmet and gloves not worn during work at {a}",
        "Contractor slipped on wet surface near {a} at {s}, minor injury, first aid given",
        "Lock-out tag-out procedure not followed fully during maintenance of {a}",
        "Ladder used without secondary support at {a}, supervisor stopped the job",
        "Worker experienced heat exhaustion during panel cleaning near {a}, rested and recovered",
        "Unauthorised person entered restricted area around {a} at {s}",
    ],
    ("safety", "high"): [
        "Near miss: vehicle reversing close to workers near {a} during night shift",
        "Electrical flash observed while opening live panel at {a}, technician moved away unhurt",
        "Fall from height near miss while climbing {a}, harness anchor point not used",
        "Fire started in {a} at {s}, extinguished with portable extinguishers, evacuation done",
        "Lost time injury: technician fractured arm when a heavy cover fell at {a}",
        "Arc fault and burn injury during energisation of {a}, worker taken to hospital",
        "Dropped object from nacelle landed near ground crew at {a}, near miss",
    ],
    ("environment", "low"): [
        "Small amount of waste packaging left outside designated bin near {a}",
        "Minor dust generation during road grading near {a}, water sprayed",
        "Noise complaint logged from nearby village about {a}, within permissible limits",
    ],
    ("environment", "medium"): [
        "Water usage spike during module cleaning due to valve leakage at {a}",
        "Used oil stored without secondary containment near {a} at {s}",
        "Vegetation cleared beyond approved boundary near {a}",
        "Wastewater overflow from cleaning tank at {a}, collected manually",
        "Excess diesel consumption recorded for {a}, suspected leakage in fuel line",
    ],
    ("environment", "high"): [
        "Oil seepage observed near {a}, spill reached soil, contained with spill kit",
        "Transformer oil leak at {a}, bund overflowed after heavy rain at {s}",
        "SF6 gas leak alarm triggered at {a}, area isolated and gas recovered",
        "Bird collision with turbine blades recorded near {a}, protected species involved",
        "Battery electrolyte leak from {a}, soil contamination suspected",
        "Hazardous waste drums found damaged and leaking near {a}",
    ],
    ("governance", "low"): [
        "Monthly safety inspection checklist for {a} submitted two days late",
        "Training attendance records incomplete for contractors at {s}",
        "Vendor safety induction register missing signatures for {a} work",
    ],
    ("governance", "medium"): [
        "Permit to work for {a} closed without supervisor countersignature at {s}",
        "Emission monitoring data for {a} not uploaded within the reporting window",
        "Waste manifest missing for disposal of scrap from {a}",
        "Contractor worked at {a} with expired medical fitness certificate",
    ],
    ("governance", "high"): [
        "Regulatory consent to operate expired for {a} at {s}, operation continued",
        "Incident at {a} not reported to authority within the mandatory timeline",
        "Falsified inspection record found for {a}, audit finding raised",
        "Statutory pressure vessel inspection overdue for {a}, work continued unauthorised",
    ],
}

rows = []
n = 1
for (cat, sev), templates in T.items():
    # more rows for common classes so the data is realistic but not uniform
    reps = {"low": 25, "medium": 30, "high": 25}[sev]
    for _ in range(reps):
        tpl = random.choice(templates)
        text = tpl.format(a=random.choice(ASSETS), s=random.choice(SITES))
        # light noise: lowercase / extra context to avoid exact duplicates
        if random.random() < 0.3:
            text += random.choice([", reported by shift in-charge", ", investigation started",
                                   ", photos attached", ", CAPA raised"])
        month = random.randint(1, 9)
        day = random.randint(1, 28)
        rows.append({
            "incident_id": f"X{n:03d}",
            "narrative": text,
            "target_category": cat,
            "target_severity": sev,
            "site": random.choice(SITES),
            "date": f"2026-{month:02d}-{day:02d}",
            "source": "synthetic",
        })
        n += 1

df = pd.DataFrame(rows).drop_duplicates("narrative").reset_index(drop=True)
OUT.parent.mkdir(exist_ok=True)
df.to_csv(OUT, index=False)
print(f"wrote {len(df)} rows to {OUT}")
print(df.groupby(["target_category", "target_severity"]).size())
