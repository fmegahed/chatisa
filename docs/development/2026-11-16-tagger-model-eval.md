# Job tagger model check: gemini-3.8-flash vs gemini-4-hydrogen

Sample: 60 fresh USAJobs postings (every 8 of 485, sorted by title), 56 tagged by all three calls. Instructions and schema as in production (v2 vocabulary). Run cost: $1.59.

| Measure | Baseline vs itself | Baseline vs candidate |
|---|---|---|
| Skill sets, mean Jaccard | 0.81 | 0.72 |
| Category agrees | 100% | 98% |
| Seniority flag agrees | 91% | 93% |
| Visa sponsorship agrees | 100% | 98% |

| Per model | Baseline | Candidate |
|---|---|---|
| Tagging failures | 1 of 120 calls | 3 of 60 calls |
| Skills per posting, mean | 8.0 | 9.8 |
| Cost per posting, mean | $0.0052 | $0.0170 |
| Seconds per posting, mean | 8.4 | 8.3 |

Candidate claims "sponsors" where the baseline never did: 0.

## Largest skill disagreements (candidate vs baseline run 1)

| Posting | Only baseline | Only candidate |
|---|---|---|
| Recreation Aid & Assistant (Lifeguard) (Seasonal) | - | communication, teamwork, leadership |
| Contract Specialist | - | technical_writing, compliance, teamwork, communication, supply_chain, stakeholder_management |
| Clinical Lab Scientist | consulting, data_analysis | compliance, problem_solving, teamwork |
| ADMINISTRATIVE PROGRAM COORDINATOR | technical_writing | project_management, stakeholder_management, research |
| Telecommunications Manager (Cyber) | stakeholder_management | security_operations, information_systems, enterprise_systems, it_management, it_strategy, consulting |
| Information Technology Specialist (Customer Support/Network  | - | it_management, communication, systems_analysis, project_management |
| Supervisory Recreation Assistant (Surf-Lifeguard) | - | communication, problem_solving |
| IT SPECIALIST (NETWORK) | - | information_systems, systems_analysis, it_management, stakeholder_management, consulting |
| IT SPECIALIST (DATA MANAGEMENT) | - | systems_analysis, sdlc, business_intelligence, stakeholder_management, consulting |
| Digital Forensic Analyst | enterprise_systems | information_systems, data_analysis, presentation_skills, stakeholder_management, business_process, statistical_analysis, mathematics |

## Failures

- gemini-3.8-flash: No object generated: could not parse the response.
- gemini-4-hydrogen: No object generated: the model did not return a response.
- gemini-4-hydrogen: No object generated: the model did not return a response.
- gemini-4-hydrogen: No object generated: could not parse the response.
