# v6.7.0 catalog review packet

For the professor's release-gate review of the FSB catalog, the business skills, the course-to-skill links, and the job tagger. Nothing here reaches students until you sign off and v6.7.0 ships.

Sources: the [Miami Bulletin 2026-27](https://bulletin.miamioh.edu/) program pages listed in `web/catalog/programs.config.json`, and the [syllabus library](https://syllabus.miamioh.edu/en-US/syllabus-library) (Fall 2026 pull; 115 of 166 courses had a syllabus). The next syllabus pull runs Feb 1, 2027.

Mapping run: 160 courses mapped by two models (claude-sonnet-5 and gpt-6-sol), $2.31. A link needs both models; where they differ on depth, the lower level is kept (181 links lowered that way). 349 suggestions from only one model were dropped. Approved links from before this release are never removed or re-levelled by the models.

## How to reply

For each section, "accept" keeps the default shown. Otherwise name the rows to change, for example "Clashes: take the models' level on 3, 7, 12".

## 1. The 30 business skills

New in the "business" category. Each is tagged on a job posting only when the posting names it in whole words (the anti-overselling guard), and a student reaches a strong level only with an anchor course, a confirmed extra, or their own override.

| # | Skill | Also matches | Courses linking it (anchor / applied / exposure) |
|---|---|---|---|
| 1 | Financial Accounting & Reporting | financial reporting, gaap, ifrs, financial statements, accounting | 9 (3 / 2 / 4) |
| 2 | Managerial & Cost Accounting | cost accounting, management accounting, budgeting, variance analysis, budget formulation, budget execution, cost analysis | 3 (1 / 1 / 1) |
| 3 | Auditing | financial audit, external audit, internal audit, assurance | 1 (1 / 0 / 0) |
| 4 | Tax | taxation, tax accounting, income tax, tax planning | 2 (1 / 1 / 0) |
| 5 | Corporate Finance | capital budgeting, cost of capital, capital structure, financial management, working capital | 6 (6 / 0 / 0) |
| 6 | Financial Analysis | financial statement analysis, ratio analysis, fp&a, financial planning and analysis | 16 (4 / 10 / 2) |
| 7 | Financial Modeling & Valuation | valuation, dcf, discounted cash flow, company valuation, financial model | 12 (6 / 6 / 0) |
| 8 | Investments & Portfolio Management | portfolio management, asset management, equity research, securities analysis, fixed income, derivatives | 13 (7 / 3 / 3) |
| 9 | Personal Financial Planning | personal finance, wealth management, retirement planning, financial advising | 1 (1 / 0 / 0) |
| 10 | Real Estate Analysis | real estate finance, property valuation, commercial real estate, real estate development | 3 (2 / 1 / 0) |
| 11 | Economic Analysis | microeconomics, macroeconomics, economic modeling, economic research | 11 (5 / 2 / 4) |
| 12 | Econometrics | panel data, instrumental variables, applied econometrics | 2 (2 / 0 / 0) |
| 13 | Marketing Strategy | marketing management, go-to-market, market positioning, marketing planning, segmentation and targeting | 13 (7 / 5 / 1) |
| 14 | Market Research | marketing research, consumer research, survey design, focus groups | 7 (3 / 4 / 0) |
| 15 | Consumer Behavior | buyer behavior, consumer psychology, shopper insights | 7 (1 / 3 / 3) |
| 16 | Digital Marketing | seo, search engine optimization, social media marketing, content marketing, email marketing, sem | 9 (3 / 4 / 2) |
| 17 | Sales | professional selling, business development, account management, sales management | 3 (2 / 1 / 0) |
| 18 | Brand Management | branding, brand strategy, advertising, integrated marketing communications | 6 (2 / 1 / 3) |
| 19 | Operations Management | production planning, capacity planning, operations planning, service operations | 4 (2 / 2 / 0) |
| 20 | Process Improvement (Lean and Six Sigma) | lean, six sigma, lean six sigma, continuous improvement, kaizen | 3 (1 / 2 / 0) |
| 21 | Human Capital Management | human resources, hr, hrm, talent management, recruiting, compensation and benefits, workforce planning | 8 (4 / 3 / 1) |
| 22 | Organizational Behavior | organizational psychology, team dynamics, change management, organizational culture | 11 (4 / 4 / 3) |
| 23 | Negotiation | negotiating, conflict resolution, deal making | 6 (3 / 1 / 2) |
| 24 | Entrepreneurship | new ventures, startups, venture creation, business planning, business model canvas | 16 (7 / 5 / 4) |
| 25 | Innovation & Design Thinking | design thinking, human-centered design, innovation management, creative problem solving | 8 (3 / 3 / 2) |
| 26 | Business Law | contract law, legal environment of business, employment law, corporate law | 7 (5 / 0 / 2) |
| 27 | International Business | global business, international trade, cross-cultural management, global markets | 7 (2 / 0 / 5) |
| 28 | Sustainability | esg, corporate social responsibility, csr, sustainable business | 6 (2 / 1 / 3) |
| 29 | Business Ethics | corporate ethics, ethical decision making, corporate governance | 9 (1 / 1 / 7) |
| 30 | Strategic Management | competitive strategy, corporate strategy, strategy execution | 1 (1 / 0 / 0) |

Default: accept all 30.

## 2. Level clashes on approved links (52, down from 57)

The course-level rule in section 8 settled 5 of the original clashes: once both sides are capped by course level, they agree.

The models proposed a level that differs from the link you approved earlier ("Models" shows both; the lower one is what the rules would use, and "Direction" compares that with yours). Default: keep your approved level. Rows marked higher are the ones most worth a look, because a higher level makes the course count for more.

| # | Course | Skill | Approved | Models | Direction |
|---|---|---|---|---|---|
| 1 | ISA 225 Principles of Business Analytics | Data Mining | exposure | applied | higher |
| 2 | ISA 225 Principles of Business Analytics | Data Visualization | applied | exposure | lower |
| 3 | ISA 225 Principles of Business Analytics | Forecasting & Time Series | exposure | applied | higher |
| 4 | ISA 241 Database for Analytics | Data Wrangling | applied | exposure / applied | lower |
| 5 | ISA 301 Business Data Communications and Security | Cybersecurity | applied | anchor | higher |
| 6 | ISA 303 Enterprise Systems | Business Process Analysis | exposure | anchor | higher |
| 7 | ISA 303 Enterprise Systems | CRM Systems | applied | exposure | lower |
| 8 | ISA 305 Information Technology Governance, Risk Management, Security and Audit | Cybersecurity | exposure | anchor | higher |
| 9 | ISA 305 Information Technology Governance, Risk Management, Security and Audit | IT Audit | applied | applied / exposure | lower |
| 10 | ISA 321 Optimization in Business Analytics | Operations Research | applied | anchor | higher |
| 11 | ISA 333 Nonparametric Statistics | Hypothesis Testing | applied | anchor | higher |
| 12 | ISA 335 Blockchain and Business Applications | Encryption & Cryptography | applied | applied / exposure | lower |
| 13 | ISA 336 Generative AI in Business | Responsible AI | applied | anchor / exposure | lower |
| 14 | ISA 345 Database Systems and Data Warehousing | Data Warehousing | applied | anchor / exposure | lower |
| 15 | ISA 365 Statistical Monitoring and Design of Experiments | A/B Testing | exposure | anchor | higher |
| 16 | ISA 365 Statistical Monitoring and Design of Experiments | Statistical Process Control | anchor | anchor / exposure | lower |
| 17 | ISA 387 Designing Business Systems | Agile Methods | exposure | anchor | higher |
| 18 | ISA 387 Designing Business Systems | Business Process Analysis | applied | exposure / applied | lower |
| 19 | ISA 387 Designing Business Systems | Project Management | exposure | applied | higher |
| 20 | ISA 391 Applied Regression Analysis in Business | Statistical Inference | applied | anchor | higher |
| 21 | ISA 401 Business Intelligence and Data Visualization | Data Warehousing | applied | exposure | lower |
| 22 | ISA 401 Business Intelligence and Data Visualization | Power BI | applied | applied / exposure | lower |
| 23 | ISA 401 Business Intelligence and Data Visualization | Tableau | applied | applied / exposure | lower |
| 24 | ISA 403 Building Web and Mobile Business Applications | Mobile Development | applied | anchor | higher |
| 25 | ISA 405 Information Security | Encryption & Cryptography | applied | exposure | lower |
| 26 | ISA 405 Information Security | Risk Management | applied | anchor | higher |
| 27 | ISA 419 Data Driven Security | Anomaly Detection | applied | anchor / exposure | lower |
| 28 | ISA 419 Data Driven Security | Cybersecurity | applied | anchor | higher |
| 29 | ISA 424 Data Infrastructure for the Enterprise | Cloud Computing | applied | applied / exposure | lower |
| 30 | ISA 424 Data Infrastructure for the Enterprise | Data Architecture | anchor | exposure / anchor | lower |
| 31 | ISA 424 Data Infrastructure for the Enterprise | Data Lakes | applied | anchor / exposure | lower |
| 32 | ISA 424 Data Infrastructure for the Enterprise | Data Warehousing | applied | anchor / exposure | lower |
| 33 | ISA 424 Data Infrastructure for the Enterprise | NoSQL Databases | applied | anchor / exposure | lower |
| 34 | ISA 491 Introduction to Data Mining in Business | Classification Models | anchor | anchor / applied | lower |
| 35 | ISA 491 Introduction to Data Mining in Business | Regression Analysis | applied | exposure / applied | lower |
| 36 | ISA 496 Business Analytics Practicum | Data Analysis | applied | anchor | higher |
| 37 | ISA 496 Business Analytics Practicum | Data Visualization | exposure | applied | higher |
| 38 | ISA 616 Communicating with Data | Communication | applied | anchor | higher |
| 39 | ISA 621 Enabling Technology Topics I | Digital Transformation | anchor | anchor / exposure | lower |
| 40 | ISA 629 Leveraging IT and Data Across the Business | Data Analysis | anchor | exposure / applied | lower |
| 41 | ISA 630 Machine Learning Applications in Business | Ensemble Methods | applied | anchor | higher |
| 42 | ISA 632 Big Data Analytics and Modern AI | Data Governance | applied | applied / exposure | lower |
| 43 | ISA 632 Big Data Analytics and Modern AI | Data Lakes | applied | anchor | higher |
| 44 | ISA 632 Big Data Analytics and Modern AI | Natural Language Processing | applied | anchor / exposure | lower |
| 45 | ISA 633 Experimental Design and Causal Methods | Experimental Design | applied | anchor | higher |
| 46 | ISA 634 Systems Modeling and Optimization | Network Analysis | applied | anchor | higher |
| 47 | ISA 634 Systems Modeling and Optimization | Operations Research | applied | anchor | higher |
| 48 | ISA 641 Data Discovery Through Business Analytics for Managers | Programming Fundamentals | applied | exposure / applied | lower |
| 49 | ISA 645 Business Analytics for the Executive | Data Analysis | applied | anchor | higher |
| 50 | ISA 645 Business Analytics for the Executive | Marketing & Customer Analytics | anchor | anchor / exposure | lower |
| 51 | ISA 650 Business Analytics Practicum | Data Analysis | applied | anchor | higher |
| 52 | ISA 650 Business Analytics Practicum | Problem Solving | applied | anchor | higher |

## 3. Courses with no anchor skill (6)

The two models did not agree on any skill for these courses, so they currently have no links and add nothing to a student's skills. (Under the section 8 rule, ACC 256 as a 200-level course would need only one evidenced link, not an anchor.) They are in scope because a program lists them as options. The skill list has no GIS or geospatial skill, which is why the GEO courses came back empty. Default: leave them unlinked. Alternative: add a "Geospatial Analysis (GIS)" skill in a later release and anchor GEO 442 to it.

- **ACC 256 Accountancy Career Exploration and Planning**: no links
- **GEO 442 Advanced Geographic Information Systems**: no links
- **GEO 451 Urban and Regional Planning**: no links
- **GEO 454 Urban Geography**: no links
- **GEO 459 Advanced Urban and Regional Planning**: no links
- **PSY 376 Psychology of Judgment, Decision Making, and Reasoning**: no links

## 4. Approved links neither model proposed (46)

Kept, because approved links are never removed by a model run. Default: keep. Name any to drop.

| # | Course | Skill | Level |
|---|---|---|---|
| 1 | ISA 211 Information Technology and Data Driven Decision Making in Business | Business Intelligence | exposure |
| 2 | ISA 211 Information Technology and Data Driven Decision Making in Business | IT Strategy | exposure |
| 3 | ISA 225 Principles of Business Analytics | Data Analysis | applied |
| 4 | ISA 235 Information Technology and the Intelligent Enterprise | Microsoft Excel | applied |
| 5 | ISA 235 Information Technology and the Intelligent Enterprise | Professional Ethics | exposure |
| 6 | ISA 242 Programming for Analytics | Python | applied |
| 7 | ISA 303 Enterprise Systems | IT Management | exposure |
| 8 | ISA 335 Blockchain and Business Applications | Cybersecurity | exposure |
| 9 | ISA 345 Database Systems and Data Warehousing | Data Wrangling | applied |
| 10 | ISA 401 Business Intelligence and Data Visualization | Dashboard Design | applied |
| 11 | ISA 403 Building Web and Mobile Business Applications | SQL | exposure |
| 12 | ISA 405 Information Security | Identity & Access Management | exposure |
| 13 | ISA 406 IT Project Management | Agile Methods | applied |
| 14 | ISA 406 IT Project Management | Risk Management | exposure |
| 15 | ISA 414 Managing Big Data | Apache Spark | applied |
| 16 | ISA 414 Managing Big Data | NoSQL Databases | applied |
| 17 | ISA 414 Managing Big Data | Natural Language Processing | exposure |
| 18 | ISA 414 Managing Big Data | Streaming Analytics | exposure |
| 19 | ISA 419 Data Driven Security | Classification Models | exposure |
| 20 | ISA 424 Data Infrastructure for the Enterprise | Amazon Web Services | exposure |
| 21 | ISA 424 Data Infrastructure for the Enterprise | Microsoft Azure | exposure |
| 22 | ISA 424 Data Infrastructure for the Enterprise | Data Governance | exposure |
| 23 | ISA 444 Business Forecasting | Predictive Modeling | applied |
| 24 | ISA 444 Business Forecasting | R | applied |
| 25 | ISA 491 Introduction to Data Mining in Business | Machine Learning | applied |
| 26 | ISA 491 Introduction to Data Mining in Business | Deep Learning | exposure |
| 27 | ISA 495 Managing the Intelligent Enterprise | Communication | applied |
| 28 | ISA 496 Business Analytics Practicum | Presentation Skills | applied |
| 29 | ISA 496 Business Analytics Practicum | Project Management | applied |
| 30 | ISA 612 Advanced Business Intelligence | Data Warehousing | exposure |
| 31 | ISA 616 Communicating with Data | R | exposure |
| 32 | ISA 621 Enabling Technology Topics I | IT Management | applied |
| 33 | ISA 628 Information Technology and Analytic's Role in the Enterprise | Data Analysis | applied |
| 34 | ISA 629 Leveraging IT and Data Across the Business | Microsoft Excel | applied |
| 35 | ISA 629 Leveraging IT and Data Across the Business | Business Process Analysis | applied |
| 36 | ISA 629 Leveraging IT and Data Across the Business | Data Visualization | exposure |
| 37 | ISA 630 Machine Learning Applications in Business | Deep Learning Frameworks | applied |
| 38 | ISA 630 Machine Learning Applications in Business | Python | applied |
| 39 | ISA 630 Machine Learning Applications in Business | Model Evaluation | applied |
| 40 | ISA 630 Machine Learning Applications in Business | Responsible AI | exposure |
| 41 | ISA 632 Big Data Analytics and Modern AI | Apache Spark | applied |
| 42 | ISA 632 Big Data Analytics and Modern AI | Deep Learning | exposure |
| 43 | ISA 641 Data Discovery Through Business Analytics for Managers | R | exposure |
| 44 | ISA 645 Business Analytics for the Executive | Business Strategy | applied |
| 45 | ISA 645 Business Analytics for the Executive | Data Visualization | exposure |
| 46 | ISA 650 Business Analytics Practicum | Presentation Skills | applied |

## 5. Job tagger v2

Details and the full spot-check table: [2026-09-24-tag-eval.md](2026-09-24-tag-eval.md). Summary: 59 federal postings tagged by both versions; existing analytics and IS tags kept 85% against a 91% run-to-run baseline; 20 new business-domain tags, all with supporting wording. Three borderline tags to rule on: Human Capital Management from a department name, Operations Management from a list of accepted degrees, Negotiation from labor relations work. Default: accept the tagger as is.

## 6. Parser gaps

The last collect run parsed all 13 programs and every course block: no unparsed rows and no in-scope course missing from the Bulletin (Independent Studies excluded by design). Nothing to decide.

## 7. Prerequisite reading, corrected after the code review

The picker adds a course's prerequisites automatically (labelled and removable). The independent review found the Bulletin's prerequisite text was misread in some cases, which would have marked courses Done that a student never took. After the fix:

- **FIN 401**: "one of ISA 225, STA 261, STA 301 or STA 368" is now one choice, not three required courses. As a four-way choice it adds nothing automatically.
- **Now treated as uncertain, so nothing is added for them**: MGT 295 and ISA 391 (a list ending in "or"); FIN 331 and FIN 381 ("A, or B and C", which has no reliable grouping); MTH 141 and MTH 151 (a placement score can replace the course); ACC 361, BUS 284 and ISA 403 (wording the parser cannot group with confidence).
- **Unchanged, now read from the Bulletin's explicit parentheses**: ISA 225 and IMS 440.

Default: accept. Name any course whose prerequisites you want the picker to add after all.


## 8. Course level caps skill depth (your rule, 2026-09-24)

100-level courses give exposure only, 200-level courses at most applied, and anchors start at 300 (graduate courses uncapped). One exception, at your call: Excel in CSE 148 stays an anchor. Applied to every link, including the ISA links approved earlier: 37 anchors became applied, 16 anchors and 14 applied links became exposure. The mapper enforces it on every future run, the models are told the rule, and a unit test fails if any link exceeds its course's cap.

What a student with only the 18 required core courses now sees: Strong in 3 skills (Business Law from BLS 342 and Corporate Finance from FIN 301, both 300-level core courses, and Excel from the CSE 148 exception), Working in 29, Introduced in 27.
