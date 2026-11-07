# ==================================================
# TASK 1: AI MILESTONES
# ==================================================

def display_ai_history():

    # Format:
    # (starting year, period, event, significance, source)

    history = [

        (
            1950,
            "1950",
            "Turing proposes the imitation game",
            "Turing reframed machine intelligence as an "
            "observable text conversation.",
            "https://academic.oup.com/mind/article/LIX/236/433/986238"
        ),

        (
            1955,
            "1955–1956",
            "Dartmouth proposal and summer project",
            "The proposal named artificial intelligence as a "
            "research program and helped establish AI as a field.",
            "https://www-formal.stanford.edu/jmc/history/dartmouth.html"
        ),

        (
            1966,
            "1966",
            "ELIZA is published",
            "ELIZA showed that conversational responses could "
            "sound understanding without proving actual understanding.",
            "https://doi.org/10.1145/365153.365168"
        ),

        (
            1974,
            "c. 1974–1980",
            "First AI winter",
            "Expectations and support cooled as ambitious AI "
            "claims encountered technical limits and skepticism.",
            "https://publications.parliament.uk/pa/ld201719/"
            "ldselect/ldai/100/10005.htm"
        ),

        (
            1980,
            "1980s",
            "Expert systems expand",
            "Rule-based expert systems brought narrow specialist "
            "decision procedures into commercial use.",
            "https://publications.parliament.uk/pa/ld201719/"
            "ldselect/ldai/100/10005.htm"
        ),

        (
            2012,
            "2012",
            "AlexNet uses ImageNet data and GPU compute",
            "A deep neural network trained on a large labeled "
            "image dataset using GPUs substantially improved "
            "image-classification results.",
            "https://papers.nips.cc/paper_files/paper/2012/hash/"
            "c399862d3b9d6b76c8436e924a68c45b-Abstract.html"
        ),

    ]

    ordered_history = sorted(
        history,
        key=lambda milestone: milestone[0]
    )

    print("\nTASK 1: AI MILESTONES\n")

    for number, milestone in enumerate(ordered_history, start=1):

        _, period, event, significance, source = milestone

        print(f"{number}. {period} — {event}")

        print(f"\nSignificance: {significance}")

        print(f"\nSource: {source}")

        print("\n" + "-" * 65 + "\n")


# ==================================================
# TASK 2: SCHOLARSHIP RULE CHECKER
# All conditions are synthetic lab assumptions.
# ==================================================

MINIMUM_SCORE = 70

MAXIMUM_INCOME = 50_000


def assess(applicant):

    checks = [

        (
            applicant["score"] >= MINIMUM_SCORE,
            "Score is below the synthetic minimum of 70."
        ),

        (
            applicant["documents_complete"],
            "Required documents are incomplete."
        ),

        (
            applicant["local_resident"],
            "LocalResidency condition is not met."
        ),

        (
            applicant["annual_income"] <= MAXIMUM_INCOME,
            "IncomeCap condition is not met."
        ),

    ]

    failed_reasons = [

        message

        for condition_met, message in checks

        if not condition_met

    ]

    status = "FAIL" if failed_reasons else "PASS"

    reason = (
        " ".join(failed_reasons)
        if failed_reasons
        else "All four explicit rules are met."
    )

    return status, reason


def run_scholarship_tests():

    # Format:
    # (name, score, documents, residency, income, expected)

    test_cases = [

        ("Case 1 — pass", 82, True, True, 42_000, "PASS"),

        (
            "Case 2 — LocalResidency failure",
            82, True, False, 42_000, "FAIL"
        ),

        (
            "Case 3 — IncomeCap failure",
            82, True, True, 50_001, "FAIL"
        ),

        (
            "Case 4 — score threshold boundary",
            70, True, True, 50_000, "PASS"
        ),

    ]

    matched_results = 0

    print("\nTASK 2: SCHOLARSHIP ASSESSMENT\n")

    for name, score, documents, resident, income, expected in test_cases:

        applicant = {

            "score": score,

            "documents_complete": documents,

            "local_resident": resident,

            "annual_income": income,

        }

        actual, reason = assess(applicant)

        test_matches = actual == expected

        matched_results += int(test_matches)

        print(name)

        print(f"\nInput: {applicant}")

        print(f"\nExpected: {expected}")

        print(f"\nActual: {actual}")

        print(f"\nReason: {reason}")

        print(
            "\nTest result:",
            "MATCH" if test_matches else "MISMATCH"
        )

        print("\n" + "-" * 65 + "\n")

    print(
        f"Summary: {matched_results}/{len(test_cases)} "
        "results matched expectations."
    )


# ==================================================
# MAIN PROGRAM
# ==================================================

def main():

    display_ai_history()

    run_scholarship_tests()


if __name__ == "__main__":

    main()