# Job tagger model check: gemini-3.8-flash vs gemini-4-hydrogen (thinking low)

Sample: 60 fresh USAJobs postings (every 8 of 487, sorted by title), 60 tagged by all three calls. Instructions and schema as in production (v2 vocabulary). Run cost: $1.27.

| Measure | Baseline vs itself | Baseline vs candidate |
|---|---|---|
| Skill sets, mean Jaccard | 0.79 | 0.75 |
| Category agrees | 100% | 100% |
| Seniority flag agrees | 93% | 90% |
| Visa sponsorship agrees | 98% | 93% |

| Per model | Baseline | Candidate |
|---|---|---|
| Tagging failures | 0 of 120 calls | 0 of 60 calls |
| Skills per posting, mean | 7.9 | 8.7 |
| Cost per posting, mean | $0.0051 | $0.0108 |
| Seconds per posting, mean | 7.9 | 4.2 |
| Thinking tokens per posting, mean | 645 | 620 |

Candidate claims "sponsors" where the baseline never did: 0.

## Largest skill disagreements (candidate vs baseline run 1)

| Posting | Only baseline | Only candidate |
|---|---|---|
| Management Analyst | information_systems, financial_accounting | systems_analysis, sdlc, business_process, business_strategy, data_analysis |
| ADMINISTRATIVE PROGRAM COORDINATOR | technical_writing | cybersecurity, project_management, stakeholder_management |
| Recreation Therapist | - | data_analysis, risk_management, compliance |
| Clinical Lab Scientist | consulting, data_analysis | teamwork, information_systems |
| Health Services Assistant | supply_chain | research |
| Supervisory HR Specialist (Recruitment and Placement) | - | data_analysis, project_management, risk_management, organizational_behavior |
| Legal Administrative Specialist | teamwork | data_analysis, consulting, mathematics, stakeholder_management |
| Instructional Systems Specialist | - | information_systems, problem_solving, communication, digital_transformation |
| Immigration Officer | risk_management, stakeholder_management, presentation_skills | - |
| Lead Human Resources Specialist (SBU) - Detail/Temporary Pro | leadership | presentation_skills, compliance, data_analysis, stakeholder_management |

