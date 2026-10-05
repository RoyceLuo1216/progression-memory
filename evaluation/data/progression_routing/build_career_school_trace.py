"""Build a fictional, fully labeled career/school progression sequence.

Human-authored routing examples define a contract, not empirical writer quality.
"""

import json
from pathlib import Path


def root(identity, scope, state, reason):
    return dict(action="create_root", identity=identity, scope=scope, state=state,
                change=state, new_segment=True, reason=reason)


def child(parent, identity, scope, state, reason):
    return dict(action="create_child", parent_id=f"progression:{parent}",
                identity=identity, scope=scope, state=state, change=state,
                new_segment=True, reason=reason)


def attach(target, state, change, reason, phase=False):
    return dict(action="attach", target_ids=[f"progression:{target}"],
                state=state, change=change, new_segment=phase, reason=reason)


def multi(updates, reason):
    return dict(action="multi_attach", target_ids=[f"progression:{p}" for p, *_ in updates],
                target_updates=[dict(progression_id=f"progression:{p}", state=s,
                                     change=c, new_segment=b) for p, s, c, b in updates],
                reason=reason)


def skip(reason):
    return dict(action="no_update", reason=reason)


# Progression ID order: education 0; university 1; year 1 2; year 2 3;
# year 3 4; research project 5; graduate applications 6; career 7;
# internship search 8; bank job 9; software-company job 10; master's 11.
ROWS = [
    ("2022-05-01", "I want to study computer science after high school.", "user",
     root("Education", "Formal learning and credentials", "Plans to study computer science after high school", "Long-term education goal")),
    ("2022-09-01", "I enrolled in a computer science degree at Northbridge University.", "user",
     child(0, "Northbridge University degree", "Undergraduate degree and milestones", "Enrolled in a computer science degree at Northbridge University", "A particular degree is a durable part of education")),
    ("2022-09-15", "My first year began with introductory programming and calculus.", "user",
     child(1, "University year one", "First academic year of the Northbridge degree", "Taking introductory programming and calculus in year one", "A bounded academic year has its own courses and milestones")),
    ("2023-01-10", "I finished introductory programming and started data structures this term.", "user",
     attach(2, "Finished introductory programming; taking data structures in year one", "Course changed within year one", "Same academic year continues")),
    ("2023-05-01", "I completed my first year of the degree.", "user",
     multi([(2, "Completed year one", "Year one completed", True),
            (1, "Completed the first year of the Northbridge degree", "Degree advanced by one year", False)],
           "Completes the year and advances its parent degree")),
    ("2023-09-01", "I began my second year at Northbridge, studying databases and statistics.", "user",
     child(1, "University year two", "Second academic year of the Northbridge degree", "Studying databases and statistics in year two", "New bounded academic year under the same degree")),
    ("2024-01-15", "I struggled with my statistics course, so I joined weekly study sessions.", "user",
     attach(3, "Taking statistics and attending weekly study sessions after struggling", "Added study sessions", "The same second-year coursework evolves")),
    ("2024-05-01", "I completed my second year at Northbridge.", "user",
     multi([(3, "Completed year two", "Year two completed", True),
            (1, "Completed two years of the Northbridge degree", "Degree advanced through year two", False)],
           "Closes the bounded year and advances the degree")),
    ("2024-09-01", "I started my third year and enrolled in machine learning.", "user",
     child(1, "University year three", "Third academic year of the Northbridge degree", "Taking machine learning in year three", "New bounded academic year under the degree")),
    ("2024-10-01", "I began a research project on conversational memory with a professor.", "user",
     child(1, "Conversational memory research", "Research project within university", "Started a conversational memory project with a professor", "Research is a distinct process that may span academic years")),
    ("2025-02-01", "Our memory project moved from lexical matching to semantic embeddings.", "user",
     attach(5, "Memory project moved from lexical matching to semantic embeddings", "Changed retrieval method", "Same research project, new technical phase", True)),
    ("2025-03-01", "I started comparing master's programs for after graduation.", "user",
     child(0, "Graduate school applications", "Selecting and applying to master's programs", "Comparing master's programs after graduation", "A new application process within education")),
    ("2025-04-01", "I shortlisted three master's programs and began drafting applications.", "user",
     attach(6, "Shortlisted three master's programs and drafting applications", "From comparison to applications", "Same admissions process moves to application phase", True)),
    ("2025-04-15", "I want software work experience before graduating.", "user",
     root("Career", "Employment and professional goals", "Wants software work experience before graduating", "New ongoing employment goal distinct from formal education")),
    ("2025-04-20", "I began applying for software internships.", "user",
     child(7, "Internship search", "Applications, interviews and offers for internships", "Applying for software internships", "A particular job search under the career goal")),
    ("2025-05-01", "I interviewed for a bank software internship.", "user",
     attach(8, "Applying and interviewing for software internships, including a bank role", "Reached interview stage", "Interview advances the same search", True)),
    ("2025-06-01", "I started a software internship at Meridian Bank.", "user",
     child(7, "Meridian Bank internship", "Employment at Meridian Bank", "Started a software internship at Meridian Bank", "A particular employer is distinct from searching for a job")),
    ("2025-08-01", "At Meridian Bank, I built an internal reporting dashboard.", "user",
     attach(9, "Working at Meridian Bank; built an internal reporting dashboard", "Built a dashboard", "Work at the same employer continues")),
    ("2025-08-15", "My friend got promoted at Meridian Bank.", "friend",
     skip("This promotion concerns a friend, not the tracked user's bank career")),
    ("2025-12-01", "My Meridian Bank internship ended, and I am looking for another software internship.", "user",
     multi([(9, "Meridian Bank internship ended", "Internship ended", True),
            (8, "Searching for another software internship after Meridian Bank", "Search resumed", True)],
           "Job ending and renewed search are separate processes for the same person")),
    ("2026-01-01", "I applied to a software internship at Harbor Software.", "user",
     attach(8, "Searching again; applied to Harbor Software", "Applied to another company", "Advances the resumed internship search")),
    ("2026-05-01", "I started my internship at Harbor Software.", "user",
     child(7, "Harbor Software internship", "Employment at Harbor Software", "Started an internship at Harbor Software", "A new employer is a distinct child of career")),
    ("2026-06-01", "At Harbor Software, I worked on the delivery prediction service.", "user",
     attach(10, "Working at Harbor Software on the delivery prediction service", "Worked on delivery predictions", "Work at the same employer continues")),
    ("2026-08-01", "Harbor Software offered me a full-time return role after my internship.", "user",
     multi([(10, "Received a full-time return offer from Harbor Software", "Internship led to a return offer", True),
            (8, "Internship search culminated in a Harbor Software return offer", "Offer received", True)],
           "The offer changes both the employer relationship and job search")),
    ("2026-08-10", "I watched a film last night.", "user",
     skip("One-off viewing provides no ongoing goal, interest or state change in this trace")),
    ("2026-09-01", "I accepted an offer to start a master's degree next year.", "user",
     multi([(6, "Accepted an offer for a master's degree starting next year", "Admissions process concluded", True),
            (0, "Plans to begin a master's degree next year after accepting an offer", "Education plan advanced", True)],
           "Acceptance concludes applications and advances the broad education plan")),
    ("2027-05-01", "I graduated from Northbridge University with my computer science degree.", "user",
     multi([(1, "Graduated from Northbridge with a computer science degree", "Degree completed", True),
            (0, "Completed the Northbridge computer science degree; master's planned", "Undergraduate degree completed", True)],
           "Degree completion updates the particular degree and broad education path")),
    ("2027-09-01", "I began the master's program in computer science.", "user",
     child(0, "Computer science master's degree", "Postgraduate study after Northbridge", "Began a master's program in computer science", "New degree is distinct from the admissions process")),
    ("2027-10-01", "In my master's program, I started a course on machine learning systems.", "user",
     attach(11, "Studying machine learning systems in the computer science master's program", "Started a course", "Coursework within the same degree")),
]


def build():
    return {"dataset": "fictional career and education timeline",
            "status": "human-authored routing contract; not real conversation training data",
            "subject_policy": "Track user; retain but do not attach third-party or incidental evidence",
            "rows": [dict(source_id=f"career-school:{i:02d}", time=t, text=text,
                          speaker_id="user", subject_id=subject, decision=decision)
                     for i, (t, text, subject, decision) in enumerate(ROWS, 1)]}


if __name__ == "__main__":
    output = Path(__file__).with_name("career_school_trace_v1.json")
    output.write_text(json.dumps(build(), indent=2) + "\n")
