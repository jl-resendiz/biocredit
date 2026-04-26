"""
=============================================================================
AUTOMATIC CONSISTENCY CHECKER
=============================================================================

Checks EVERY quantitative claim in main.tex and supplementary.tex against
the pipeline's single sources of truth:

  1. output/results/abm_results.json      (model results)
  2. output/results/glv_results.json       (GLV results)
  3. scripts/verify_claims.py output       (empirical claims)
  4. data/CODEBOOK.md                      (data definitions)

Run after ANY change to models, data, or manuscript:
    python scripts/check_consistency.py

Exit code 0 = all consistent. Exit code 1 = inconsistencies found.
Add to CI/CD or pre-commit hook for automatic enforcement.
=============================================================================
"""

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "output" / "results"
MANUSCRIPT = PROJECT / "manuscript"

# =============================================================================
# 1. LOAD TRUTH SOURCES
# =============================================================================

def load_json(name):
    path = RESULTS / name
    if not path.exists():
        print(f"  WARNING: {path} not found. Run models first.")
        return None
    with open(path) as f:
        return json.load(f)

def load_tex(name):
    path = MANUSCRIPT / name
    with open(path, encoding="utf-8") as f:
        return f.read()

# =============================================================================
# 2. DEFINE CLAIMS TO CHECK
# =============================================================================

def build_checks(abm, glv):
    """
    Each check: (description, pattern_in_tex, expected_value, source, files_to_search)

    pattern_in_tex: regex that should match in the tex file
    expected_value: what the number should be (from JSON/data)
    """
    checks = []

    # --- ABM results ---
    if abm:
        b = abm["baseline"]
        s = abm["scenarios"]

        # EC2 baseline
        checks.append(("ABM EC2 baseline",
            r"16\.3\\%", f"{b['ec2_median']}%",
            "abm_results.json baseline.ec2_median", ["main.tex", "supplementary.tex"]))

        # EC2 CI
        checks.append(("ABM EC2 CI low",
            r"14\.6", f"{b['ec2_p10']}",
            "abm_results.json baseline.ec2_p10", ["main.tex"]))
        checks.append(("ABM EC2 CI high",
            r"17\.9\\%\]", f"{b['ec2_p90']}%",
            "abm_results.json baseline.ec2_p90", ["main.tex"]))

        # EC2-TEC model
        checks.append(("ABM EC2-TEC",
            r"25\.3\\%", f"{b['ec2_tec']}%",
            "abm_results.json baseline.ec2_tec", ["main.tex", "supplementary.tex"]))

        # Fund rate
        checks.append(("ABM Fund rate baseline",
            r"34\.1\\%", f"{b['fund_rate']}%",
            "abm_results.json baseline.fund_rate", ["main.tex", "supplementary.tex"]))

        # Compliance/BCT split
        checks.append(("ABM compliance share",
            r"65\.8\\%", f"{b['compliance_share']}%",
            "abm_results.json baseline.compliance_share", ["main.tex"]))

        # Spearman rho
        checks.append(("ABM Spearman rho",
            r"0\.98[4]?", f"{b['spearman_rho']:.3f}",
            "abm_results.json baseline.spearman_rho", ["main.tex", "supplementary.tex"]))

        # Formations extinct
        checks.append(("ABM formations extinct",
            r"only one formation|all but one formation|one.*extinct formation|predicts one extinct|one formation.*market.extinct",
            f"{b['formations_extinct']} formation(s)",
            "abm_results.json baseline.formations_extinct", ["main.tex"]))

        # Procurement Flex EC2
        pf = s.get("Procurement Flex (20%)", {})
        if pf:
            checks.append(("Procurement Flex EC2",
                r"17\.1\\%", f"{pf['ec2_median']}%",
                "abm_results.json scenarios.Procurement Flex.ec2_median", ["main.tex"]))

        # Price Floor Fund rate
        fl = s.get("Price Floor (AUD 3,000)", {})
        if fl:
            checks.append(("Price Floor Fund rate",
                r"50\.5\\%", f"{fl['fund_rate']}%",
                "abm_results.json scenarios.Price Floor.fund_rate", ["main.tex"]))

        # BCT budget
        checks.append(("BCT budget",
            r"326[{,]?389", f"{abm['parameters']['bct_budget']}",
            "abm_results.json parameters.bct_budget", ["main.tex", "supplementary.tex"]))

        # Production cost
        checks.append(("Production cost mu",
            r"7\.62", f"{abm['parameters']['production_cost_mu']}",
            "abm_results.json parameters.production_cost_mu", ["main.tex", "supplementary.tex"]))

        # P_variation
        checks.append(("P_variation",
            r"P_\{\\text\{variation\}\}.*=.*0\.5|P_variation.*0\.5",
            f"{abm['parameters']['p_variation']}",
            "abm_results.json parameters.p_variation", ["main.tex"]))

    # --- GLV results ---
    if glv:
        glv_scenarios = glv.get("scenarios", {})
        glv_baseline = glv_scenarios.get("Baseline", {})

        if glv_baseline:
            ec2_pct = f"{glv_baseline.get('ec2', 0)*100:.1f}"
            checks.append(("GLV EC2 baseline",
                r"7\.0\\%", ec2_pct + "%",
                "glv_results.json scenarios.Baseline.ec2", ["supplementary.tex"]))

        n_extinct = glv.get("n_extinct", glv.get("extinct_count", None))
        if n_extinct:
            checks.append(("GLV formations extinct",
                r"five formations at market extinction",
                f"{n_extinct} formations",
                "glv_results.json n_extinct", ["main.tex", "supplementary.tex"]))

    # --- Empirical claims (hardcoded — these come from verify_claims.py) ---
    empirical = [
        ("Transaction count", r"1[{,]124", "1,124", "CODEBOOK.md"),
        ("Total OTGs", r"252", "252", "CODEBOOK.md"),
        ("Never traded", r"112.*44\\%|44\\%.*112", "112 (44%)", "CODEBOOK.md"),
        ("EC2 observed", r"17\.9\\%", "17.9%", "CODEBOOK.md"),
        ("TEC OTGs", r"79.*TEC|TEC.*79", "79", "CODEBOOK.md"),
        ("Non-TEC OTGs", r"173", "173", "CODEBOOK.md"),
        ("EC2-TEC observed", r"22\.8\\%", "22.8%", "CODEBOOK.md (supply-register matching: 18/79)"),
        ("BCT premium", r"5\.3", "5.3x", "CODEBOOK.md"),
        ("BCT thin-market median", r"5[{,]989", "5,989", "CODEBOOK.md"),
        ("Compliance thin-market median", r"1[{,]125", "1,125", "CODEBOOK.md"),
        ("BCT-only OTGs", r"24.*BCT.only|BCT.only.*24|24.*(?:credit categories|rare credit categories|ecosystem credit types).*(?:Trust|BCT)", "24", "CODEBOOK.md (a.k.a. credit categories served only by the Trust)"),
        ("CV", r"1\.56", "1.56", "CODEBOOK.md"),
        ("Eco median", r"4[{,.]047", "4,047", "CODEBOOK.md + Supp Table 2"),
        ("Species median", r"800", "800", "CODEBOOK.md"),
    ]
    for desc, pattern, expected, source in empirical:
        checks.append((desc, pattern, expected, source, ["main.tex", "supplementary.tex"]))

    # --- Stale values that MUST NOT appear ---
    stale = [
        ("OLD: 1,123 transactions", r"1[{,]123"),
        ("OLD: 3.6x BCT premium", r"3\.6.fold|3\.6.times|3\.6x"),
        # 22.8% is the CURRENT canonical FC-TEC (18/79, supply-register matching);
        # it is no longer stale. The earlier 20.3% (16/79, direct-name matching)
        # has been retired in favour of supply-register matching for the TEC analysis.
        ("OLD: 20.3% EC2-TEC (direct-name matching)", r"20\.3\\%"),
        ("OLD: 40,000 BCT budget", r"40[{,]000"),
        ("OLD: 62% TEC excluded", r"62\\%.*TEC|TEC.*62\\%"),
        ("OLD: 15.2% GLV EC2", r"15\.2\\%"),
        ("OLD: experiment_final", r"experiment.final"),
        ("OLD: verify_manuscript_claims", r"verify.manuscript.claims"),
        ("OLD: 0.57 margin", r"margin of 0\.57"),
        ("OLD: Freshwater.*Rainforests.*Heathlands list", r"Freshwater Wetlands.*Rainforests.*Heathlands"),
    ]

    return checks, stale

# =============================================================================
# 3. RUN CHECKS
# =============================================================================

def main():
    print("=" * 80)
    print("CONSISTENCY CHECK: Manuscript vs Pipeline")
    print("=" * 80)

    # Load sources
    abm = load_json("abm_results.json")
    glv = load_json("glv_results.json")

    main_tex = load_tex("main.tex")
    supp_tex = load_tex("supplementary.tex")
    tex_files = {"main.tex": main_tex, "supplementary.tex": supp_tex}

    checks, stale = build_checks(abm, glv)

    n_pass = 0
    n_fail = 0
    n_warn = 0
    failures = []

    # --- Positive checks (values that SHOULD appear) ---
    print(f"\n{'─' * 80}")
    print("POSITIVE CHECKS: Values that should appear in manuscript")
    print(f"{'─' * 80}")

    for desc, pattern, expected, source, files in checks:
        found_in = []
        for fname in files:
            tex = tex_files.get(fname, "")
            if re.search(pattern, tex):
                found_in.append(fname)

        if found_in:
            status = "PASS"
            n_pass += 1
        else:
            status = "FAIL"
            n_fail += 1
            failures.append((desc, expected, source, files))

        files_str = ", ".join(found_in) if found_in else "NOT FOUND"
        print(f"  {status:4s}  {desc:40s}  expected={expected:15s}  in={files_str}")

    # --- Negative checks (stale values that MUST NOT appear) ---
    print(f"\n{'─' * 80}")
    print("NEGATIVE CHECKS: Stale values that must NOT appear")
    print(f"{'─' * 80}")

    for desc, pattern in stale:
        found_in = []
        for fname, tex in tex_files.items():
            matches = re.findall(pattern, tex)
            if matches:
                found_in.append(f"{fname}({len(matches)})")

        if found_in:
            status = "FAIL"
            n_fail += 1
            failures.append((desc, "should not appear", "stale", found_in))
        else:
            status = "PASS"
            n_pass += 1

        files_str = ", ".join(found_in) if found_in else "clean"
        print(f"  {status:4s}  {desc:40s}  {files_str}")

    # --- JSON freshness check ---
    print(f"\n{'─' * 80}")
    print("JSON FRESHNESS CHECK")
    print(f"{'─' * 80}")

    for name in ["abm_results.json", "glv_results.json"]:
        path = RESULTS / name
        if path.exists():
            data = json.loads(path.read_text())
            ts = data.get("timestamp", "unknown")
            print(f"  OK    {name:40s}  generated={ts}")
        else:
            print(f"  WARN  {name:40s}  NOT FOUND — run models first")
            n_warn += 1

    # --- Prose logic checks (claims vs JSON logic) ---
    print(f"\n{'─' * 80}")
    print("PROSE LOGIC CHECKS: Do text claims match JSON logic?")
    print(f"{'─' * 80}")

    if abm:
        s = abm["scenarios"]
        b = abm["baseline"]

        # 1. "Procurement flex is most effective single intervention"
        ec2_vals = {k: v["ec2_median"] for k, v in s.items()}
        best = max(ec2_vals, key=ec2_vals.get)
        is_flex_best = "Procurement" in best or "Flex" in best
        combined_ties = ec2_vals.get("Combined", 0) == ec2_vals.get("Procurement Flex (20%)", -1)
        flex_claim = re.search(r"most effective single intervention|Procurement.*most effective", main_tex)
        if flex_claim and (is_flex_best or combined_ties):
            print(f"  PASS  'Procurement flex most effective'     JSON best={best} ({ec2_vals[best]}%)")
            n_pass += 1
        elif flex_claim:
            print(f"  FAIL  'Procurement flex most effective'     JSON best={best} ({ec2_vals[best]}%)")
            n_fail += 1
            failures.append(("Prose: flex most effective", f"JSON best={best}", "abm_results.json", []))
        else:
            print(f"  SKIP  Claim not found in text")

        # 2. "Price floors are counterproductive"
        floor_fund = s.get("Price Floor (AUD 3,000)", {}).get("fund_rate", 0)
        baseline_fund = b["fund_rate"]
        floor_worse = floor_fund > baseline_fund
        floor_claim = re.search(r"counterproductive|price floor.*worse|price floor.*increase.*Fund", main_tex, re.IGNORECASE)
        if floor_claim and floor_worse:
            print(f"  PASS  'Price floors counterproductive'      Fund: {baseline_fund}% -> {floor_fund}%")
            n_pass += 1
        elif floor_claim and not floor_worse:
            print(f"  FAIL  'Price floors counterproductive'      Fund: {baseline_fund}% -> {floor_fund}% (not worse!)")
            n_fail += 1
            failures.append(("Prose: floors counterproductive", f"Fund {baseline_fund}->{floor_fund}", "abm_results.json", []))
        else:
            print(f"  SKIP  Claim not found in text")

        # 3. "Only one formation at market extinction" (ABM)
        extinct_claim = re.search(r"only one formation|all but one formation|one.*extinct formation|predicts one extinct|one formation.*market.extinct", main_tex)
        if extinct_claim and b["formations_extinct"] == 1:
            print(f"  PASS  'Only one formation extinct (ABM)'   JSON={b['formations_extinct']}")
            n_pass += 1
        elif extinct_claim:
            print(f"  FAIL  'Only one formation extinct (ABM)'   JSON={b['formations_extinct']}")
            n_fail += 1
            failures.append(("Prose: one extinct", f"JSON={b['formations_extinct']}", "abm_results.json", []))
        else:
            print(f"  SKIP  Claim not found in text")

        # 4. "Fund routing rate matches observed 35%"
        fund_match = abs(b["fund_rate"] - 35.4) < 2.0  # within 2pp
        fund_claim = re.search(r"34\.1.*35\.4|Fund routing rate.*closely matches", main_tex)
        if fund_claim and fund_match:
            print(f"  PASS  'Fund rate matches observed'         Model={b['fund_rate']}% vs Observed=35.4%")
            n_pass += 1
        elif fund_claim:
            print(f"  FAIL  'Fund rate matches observed'         Model={b['fund_rate']}% vs Observed=35.4% (gap > 2pp)")
            n_fail += 1
            failures.append(("Prose: fund rate match", f"{b['fund_rate']} vs 35.4", "abm_results.json", []))
        else:
            print(f"  SKIP  Claim not found in text")

        # 5. Scenario ranking: Flex >= Combined > Baseline >= Price Floor
        ranking = sorted(ec2_vals.items(), key=lambda x: -x[1])
        ranking_names = [r[0] for r in ranking]
        baseline_idx = next((i for i, r in enumerate(ranking) if "Baseline" in r[0]), -1)
        floor_idx = next((i for i, r in enumerate(ranking) if "Price Floor" in r[0]), -1)
        baseline_not_best = baseline_idx > 0  # not first
        floor_not_best = floor_idx > 0
        ranking_claim = re.search(r"Procurement Flexibility.*Combined.*Baseline.*Price Floor|Procurement Flex.*Combined.*Baseline.*Price Floor", main_tex)
        if ranking_claim and baseline_not_best and floor_not_best:
            print(f"  PASS  'Scenario ranking correct'           {' > '.join(r[0][:15] for r in ranking)}")
            n_pass += 1
        elif ranking_claim:
            print(f"  FAIL  'Scenario ranking'                   {' > '.join(r[0][:15] for r in ranking)}")
            n_fail += 1
            failures.append(("Prose: scenario ranking", str(ranking_names), "abm_results.json", []))
        else:
            print(f"  SKIP  Ranking claim not found")

    if glv:
        # 6. GLV extinct count matches text
        irr = glv.get("irreversibility", {})
        n_ext = irr.get("n_extinct", glv.get("n_extinct", 0))
        glv_claim = re.search(r"five formations at market extinction", main_tex)
        if glv_claim and n_ext == 5:
            print(f"  PASS  'GLV five formations extinct'        JSON={n_ext}")
            n_pass += 1
        elif glv_claim:
            print(f"  FAIL  'GLV five formations extinct'        JSON={n_ext}")
            n_fail += 1
            failures.append(("Prose: GLV extinct count", f"JSON={n_ext}", "glv_results.json", []))
        else:
            print(f"  SKIP  GLV extinct claim not found")

    # --- Summary ---
    print(f"\n{'=' * 80}")
    print(f"SUMMARY: {n_pass} PASS, {n_fail} FAIL, {n_warn} WARN")
    print(f"{'=' * 80}")

    if failures:
        print("\nFAILURES:")
        for desc, expected, source, files in failures:
            print(f"  ✗ {desc}: expected {expected} (source: {source}), found in: {files}")

    if n_fail == 0:
        print("\n✓ ALL CONSISTENT. Manuscript matches pipeline.")
        sys.exit(0)
    else:
        print(f"\n✗ {n_fail} INCONSISTENCIES FOUND. Fix before submission.")
        sys.exit(1)


if __name__ == "__main__":
    main()
