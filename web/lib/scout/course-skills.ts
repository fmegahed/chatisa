/**
 * Course-to-skill mapping against lib/scout/taxonomy.ts. Initial mapping
 * authored by Claude Fable 5 in-session (2026-07-28) from the bulletin
 * descriptions, replacing careerbridge's flat free-text lists; the
 * instructor-facing review table lives at
 * docs/development/2026-07-28-course-skills-review.md and
 * scripts/generate-course-skills.mjs can regenerate this file when the
 * curriculum changes.
 *
 * Levels (design §2.2): anchor = graded deliverables demonstrate it;
 * applied = used repeatedly as a working tool; exposure = introduced.
 * Evidence phrases are written in the student's voice fragments ("built...",
 * "designed...") because they feed grounded resume bullets downstream.
 *
 * Where the bulletin does not name a tool, none is claimed: a wrong tool
 * here becomes a wrong line on a resume. The two deliberate exceptions
 * (Python in 242/381/419/630, R in 444/616) reflect how FSB actually
 * teaches those courses and are flagged for instructor review.
 */

export type CourseSkillLevel = "anchor" | "applied" | "exposure";

export interface CourseSkillLink {
  course: string;
  skillId: string;
  level: CourseSkillLevel;
  evidence?: string;
}

const L = (
  course: string,
  skillId: string,
  level: CourseSkillLevel,
  evidence?: string,
): CourseSkillLink => ({ course, skillId, level, evidence });

export const COURSE_SKILLS: CourseSkillLink[] = [
  // ISA 125 — Introduction to Business Statistics
  L("ISA 125", "statistical_analysis", "anchor", "summarized business data and described relationships among variables"),
  L("ISA 125", "statistical_inference", "anchor", "conducted one- and two-sample statistical inference"),
  L("ISA 125", "probability", "applied"),
  L("ISA 125", "hypothesis_testing", "applied"),
  L("ISA 125", "data_analysis", "applied"),

  // ISA 211 — IT and Data Driven Decision Making (non-majors)
  L("ISA 211", "information_systems", "anchor", "analyzed how organizations use information systems for data-driven decisions"),
  L("ISA 211", "data_analysis", "applied"),
  L("ISA 211", "business_intelligence", "exposure"),
  L("ISA 211", "it_strategy", "exposure"),

  // ISA 225 — Principles of Business Analytics
  L("ISA 225", "data_analysis", "anchor", "analyzed real business data end to end and communicated results"),
  L("ISA 225", "regression", "applied", "built predictive regression models on business data"),
  L("ISA 225", "data_visualization", "applied"),
  L("ISA 225", "statistical_inference", "applied"),
  L("ISA 225", "forecasting", "exposure"),
  L("ISA 225", "data_mining", "exposure"),
  L("ISA 225", "classification", "exposure"),

  // ISA 235 — IT and the Intelligent Enterprise
  L("ISA 235", "information_systems", "anchor", "evaluated the strategic role of IT and data in business transformation"),
  L("ISA 235", "it_strategy", "applied"),
  L("ISA 235", "excel", "applied", "solved data-driven business problems with spreadsheets"),
  L("ISA 235", "data_visualization", "applied"),
  L("ISA 235", "digital_transformation", "applied"),
  L("ISA 235", "generative_ai", "exposure"),
  L("ISA 235", "ethics", "exposure"),

  // ISA 241 — Database for Analytics (1.5 cr)
  L("ISA 241", "sql", "anchor", "wrote SQL to collect, manipulate, and manage structured data"),
  L("ISA 241", "database_design", "anchor", "designed logical and physical database schemas with ER modeling"),
  L("ISA 241", "data_wrangling", "applied"),

  // ISA 242 — Programming for Analytics (1.5 cr)
  L("ISA 242", "programming_fundamentals", "anchor", "wrote structured programs to access and process business data"),
  L("ISA 242", "python", "applied", "used Python libraries for data retrieval and presentation"),
  L("ISA 242", "data_wrangling", "applied"),

  // ISA 250 — Basic Math for Analytics
  L("ISA 250", "mathematics", "anchor", "applied matrix algebra, calculus, and functions to analytics problems"),
  L("ISA 250", "optimization", "exposure"),

  // ISA 301 — Business Data Communications and Security
  L("ISA 301", "networking", "anchor", "analyzed network architectures and data communications standards"),
  L("ISA 301", "cybersecurity", "applied", "evaluated network and data security threats and protections"),
  L("ISA 301", "cloud_computing", "exposure"),

  // ISA 303 — Enterprise Systems
  L("ISA 303", "enterprise_systems", "anchor", "worked with ERP, supply chain, and CRM systems and their implementation trade-offs"),
  L("ISA 303", "supply_chain", "applied"),
  L("ISA 303", "crm", "applied"),
  L("ISA 303", "business_process", "exposure"),
  L("ISA 303", "it_management", "exposure"),

  // ISA 305 — IT Governance, Risk, Security and Audit
  L("ISA 305", "it_governance", "anchor", "formulated IT governance and assurance strategy"),
  L("ISA 305", "risk_management", "anchor", "assessed IT risk and selected controls"),
  L("ISA 305", "it_audit", "applied"),
  L("ISA 305", "compliance", "applied"),
  L("ISA 305", "cybersecurity", "exposure"),

  // ISA 321 — Optimization in Business Analytics
  L("ISA 321", "optimization", "anchor", "built linear, integer, and nonlinear programs for production, supply chain, and finance decisions"),
  L("ISA 321", "operations_research", "applied"),
  L("ISA 321", "network_analysis", "applied"),

  // ISA 333 — Nonparametric Statistics
  L("ISA 333", "nonparametric_statistics", "anchor", "applied chi-square, runs, and association tests when distributions are unknown"),
  L("ISA 333", "hypothesis_testing", "applied"),
  L("ISA 333", "statistical_inference", "applied"),

  // ISA 335 — Blockchain and Business Applications
  L("ISA 335", "blockchain", "anchor", "analyzed blockchain applications from cryptocurrencies to smart contracts and DeFi"),
  L("ISA 335", "encryption", "applied"),
  L("ISA 335", "cybersecurity", "exposure"),

  // ISA 336 — Generative AI in Business
  L("ISA 336", "generative_ai", "anchor", "applied large language models to business content creation and process automation"),
  L("ISA 336", "prompt_engineering", "applied", "designed prompts and multimodal workflows"),
  L("ISA 336", "llm_applications", "applied", "integrated generative AI with external data"),
  L("ISA 336", "ai_ethics", "applied"),

  // ISA 345 — Database Systems and Data Warehousing
  L("ISA 345", "sql", "anchor", "implemented databases and queried them with SQL"),
  L("ISA 345", "database_design", "anchor", "designed databases with formal design methodologies"),
  L("ISA 345", "data_warehousing", "applied"),
  L("ISA 345", "data_wrangling", "applied"),

  // ISA 365 — Statistical Monitoring and Design of Experiments
  L("ISA 365", "statistical_process_control", "anchor", "monitored process data and data streams with control methods"),
  L("ISA 365", "experimental_design", "anchor", "designed experiments for business analytics questions"),
  L("ISA 365", "hypothesis_testing", "applied"),
  L("ISA 365", "ab_testing", "exposure"),

  // ISA 381 — Concepts in Business Programming
  L("ISA 381", "python", "anchor", "developed scalable data-driven applications with custom libraries"),
  L("ISA 381", "object_oriented_programming", "anchor", "designed object-oriented programs with functions and lambda expressions"),
  L("ISA 381", "data_wrangling", "applied", "processed TXT, CSV, Excel, and database data programmatically"),

  // ISA 387 — Designing Business Systems
  L("ISA 387", "systems_analysis", "anchor", "planned and evaluated business software acquisition options"),
  L("ISA 387", "sdlc", "anchor", "applied the application life cycle from requirements to delivery"),
  L("ISA 387", "business_process", "applied"),
  L("ISA 387", "agile", "exposure"),
  L("ISA 387", "project_management", "exposure"),

  // ISA 391 — Applied Regression Analysis in Business
  L("ISA 391", "regression", "anchor", "built and evaluated explanatory and predictive multiple regression models"),
  L("ISA 391", "predictive_modeling", "applied"),
  L("ISA 391", "hypothesis_testing", "applied"),
  L("ISA 391", "model_evaluation", "applied"),
  L("ISA 391", "statistical_inference", "applied"),

  // ISA 401/501 — Business Intelligence and Data Visualization
  L("ISA 401", "business_intelligence", "anchor", "built business intelligence solutions from data gathering to use"),
  L("ISA 401", "data_visualization", "anchor", "designed visualizations that communicate business insight"),
  L("ISA 401", "tableau", "applied"),
  L("ISA 401", "power_bi", "applied"),
  L("ISA 401", "dashboard_design", "applied"),
  L("ISA 401", "data_warehousing", "applied"),
  L("ISA 401", "data_mining", "exposure"),
  L("ISA 401", "etl", "exposure"),

  // ISA 403 — Building Web and Mobile Business Applications
  L("ISA 403", "web_development", "anchor", "delivered scalable web applications with client and server-side technologies"),
  L("ISA 403", "mobile_development", "applied"),
  L("ISA 403", "javascript", "applied"),
  L("ISA 403", "api_development", "applied"),
  L("ISA 403", "sql", "exposure"),

  // ISA 405 — Information Security
  L("ISA 405", "cybersecurity", "anchor", "analyzed threats, vulnerabilities, and controls with security tools and cases"),
  L("ISA 405", "risk_management", "applied", "applied risk management frameworks"),
  L("ISA 405", "compliance", "applied"),
  L("ISA 405", "it_governance", "applied"),
  L("ISA 405", "encryption", "applied"),
  L("ISA 405", "business_continuity", "exposure"),
  L("ISA 405", "identity_access_management", "exposure"),
  L("ISA 405", "ethics", "exposure"),

  // ISA 406 — IT Project Management
  L("ISA 406", "project_management", "anchor", "managed IT projects with professional techniques and software tools"),
  L("ISA 406", "agile", "applied"),
  L("ISA 406", "risk_management", "exposure"),
  L("ISA 406", "it_management", "exposure"),

  // ISA 414/514 — Managing Big Data
  L("ISA 414", "big_data", "anchor", "developed big data solutions for unstructured and large-scale datasets"),
  L("ISA 414", "spark", "applied"),
  L("ISA 414", "nosql", "applied"),
  L("ISA 414", "nlp", "exposure"),
  L("ISA 414", "hadoop", "exposure"),
  L("ISA 414", "streaming_analytics", "exposure"),

  // ISA 419 — Data Driven Security
  L("ISA 419", "security_operations", "anchor", "built intrusion-detection and phishing-detection analyses on real security logs"),
  L("ISA 419", "python", "applied", "programmed security analytics on real datasets"),
  L("ISA 419", "machine_learning", "applied"),
  L("ISA 419", "anomaly_detection", "applied", "discovered malicious patterns in security software logs"),
  L("ISA 419", "cybersecurity", "applied"),
  L("ISA 419", "classification", "exposure"),

  // ISA 424 — Data Infrastructure for the Enterprise
  L("ISA 424", "data_architecture", "anchor", "evaluated enterprise data infrastructure options for decision making"),
  L("ISA 424", "data_warehousing", "applied"),
  L("ISA 424", "data_lakes", "applied"),
  L("ISA 424", "nosql", "applied"),
  L("ISA 424", "cloud_computing", "applied"),
  L("ISA 424", "aws", "exposure"),
  L("ISA 424", "azure", "exposure"),
  L("ISA 424", "data_governance", "exposure"),

  // ISA 444/544 — Business Forecasting
  L("ISA 444", "forecasting", "anchor", "built and evaluated Box-Jenkins, exponential smoothing, and decomposition forecasts on business time series"),
  L("ISA 444", "regression", "applied", "fit time series regressions with autocorrelated errors"),
  L("ISA 444", "predictive_modeling", "applied"),
  L("ISA 444", "r", "applied", "implemented forecasting workflows in R"),
  L("ISA 444", "model_evaluation", "exposure"),

  // ISA 491/591 — Introduction to Data Mining in Business
  L("ISA 491", "data_mining", "anchor", "mined large business datasets with current software"),
  L("ISA 491", "classification", "anchor", "built tree, logistic regression, and neural net classifiers"),
  L("ISA 491", "clustering", "applied", "segmented data with cluster and market basket analysis"),
  L("ISA 491", "machine_learning", "applied"),
  L("ISA 491", "model_evaluation", "applied"),
  L("ISA 491", "regression", "applied"),
  L("ISA 491", "deep_learning", "exposure"),

  // ISA 495 — Managing the Intelligent Enterprise
  L("ISA 495", "it_strategy", "anchor", "researched a company's information systems strategy and presented findings"),
  L("ISA 495", "research", "applied"),
  L("ISA 495", "business_strategy", "applied"),
  L("ISA 495", "presentation_skills", "applied"),
  L("ISA 495", "communication", "applied"),

  // ISA 496 — Business Analytics Practicum
  L("ISA 496", "consulting", "anchor", "delivered analytics consulting to a real business client"),
  L("ISA 496", "data_analysis", "applied", "solved a client's data-driven problem end to end"),
  L("ISA 496", "stakeholder_management", "applied"),
  L("ISA 496", "presentation_skills", "applied"),
  L("ISA 496", "project_management", "applied"),
  L("ISA 496", "teamwork", "applied"),
  L("ISA 496", "data_visualization", "exposure"),

  // ISA 612 — Advanced Business Intelligence
  L("ISA 612", "business_intelligence", "anchor", "modeled structured data for business intelligence"),
  L("ISA 612", "dashboard_design", "anchor", "built interactive dashboards with charts, filters, and KPIs"),
  L("ISA 612", "data_wrangling", "applied", "retrieved, cleaned, and manipulated data for analysis"),
  L("ISA 612", "data_visualization", "applied"),
  L("ISA 612", "data_warehousing", "exposure"),

  // ISA 616 — Communicating with Data
  L("ISA 616", "data_storytelling", "anchor", "developed a data analytic product from client consultation to presentation"),
  L("ISA 616", "technical_writing", "anchor", "wrote professional white papers and technical reports"),
  L("ISA 616", "presentation_skills", "applied"),
  L("ISA 616", "data_visualization", "applied"),
  L("ISA 616", "communication", "applied"),
  L("ISA 616", "stakeholder_management", "applied"),
  L("ISA 616", "r", "exposure"),
  L("ISA 616", "ethics", "exposure"),

  // ISA 621 — Enabling Technology Topics I
  L("ISA 621", "digital_transformation", "anchor", "evaluated emerging IT and innovation methodologies in the organization"),
  L("ISA 621", "it_management", "applied"),
  L("ISA 621", "information_systems", "applied"),
  L("ISA 621", "business_process", "exposure"),

  // ISA 628 — IT and Analytics' Role in the Enterprise (1.5 cr)
  L("ISA 628", "information_systems", "anchor", "examined how IT reinvents processes and data consumption for decisions"),
  L("ISA 628", "data_analysis", "applied"),
  L("ISA 628", "it_strategy", "exposure"),
  L("ISA 628", "digital_transformation", "exposure"),

  // ISA 629 — Leveraging IT and Data Across the Business (1.5 cr)
  L("ISA 629", "data_analysis", "anchor", "applied data manipulation techniques to discipline-specific business problems"),
  L("ISA 629", "excel", "applied"),
  L("ISA 629", "business_process", "applied"),
  L("ISA 629", "data_visualization", "exposure"),

  // ISA 630 — Machine Learning Applications in Business
  L("ISA 630", "machine_learning", "anchor", "built supervised and unsupervised models for business applications"),
  L("ISA 630", "ensemble_methods", "applied", "built and customized ensemble models"),
  L("ISA 630", "deep_learning", "applied"),
  L("ISA 630", "deep_learning_frameworks", "applied"),
  L("ISA 630", "python", "applied"),
  L("ISA 630", "model_evaluation", "applied"),
  L("ISA 630", "ai_ethics", "exposure"),

  // ISA 632 — Big Data Analytics and Modern AI
  L("ISA 632", "big_data", "anchor", "integrated data sources into a lake and ran distributed analytics"),
  L("ISA 632", "spark", "applied", "used in-memory cluster computing for advanced analytics"),
  L("ISA 632", "nlp", "applied"),
  L("ISA 632", "streaming_analytics", "applied"),
  L("ISA 632", "data_lakes", "applied"),
  L("ISA 632", "data_governance", "applied"),
  L("ISA 632", "deep_learning", "exposure"),
  L("ISA 632", "computer_vision", "exposure"),
  L("ISA 632", "network_analysis", "exposure"),

  // ISA 633 — Experimental Design and Causal Methods
  L("ISA 633", "causal_inference", "anchor", "modeled causal relationships when experimentation is infeasible"),
  L("ISA 633", "ab_testing", "anchor", "designed and analyzed A/B tests and complex business experiments"),
  L("ISA 633", "experimental_design", "applied"),
  L("ISA 633", "hypothesis_testing", "applied"),
  L("ISA 633", "regression", "exposure"),

  // ISA 634 — Systems Modeling and Optimization
  L("ISA 634", "optimization", "anchor", "created mathematical programming models under real business constraints"),
  L("ISA 634", "network_analysis", "applied", "modeled network systems with graph-theoretic methods"),
  L("ISA 634", "operations_research", "applied"),
  L("ISA 634", "simulation", "exposure"),

  // ISA 641 — Data Discovery for Managers (2 cr)
  L("ISA 641", "data_analysis", "anchor", "applied programming concepts to data summarization and discovery"),
  L("ISA 641", "data_visualization", "applied"),
  L("ISA 641", "programming_fundamentals", "applied"),
  L("ISA 641", "r", "exposure"),
  L("ISA 641", "python", "exposure"),

  // ISA 645 — Business Analytics for the Executive
  L("ISA 645", "marketing_analytics", "anchor", "applied the analytics measures leading organizations use for marketing decisions"),
  L("ISA 645", "data_analysis", "applied"),
  L("ISA 645", "business_strategy", "applied"),
  L("ISA 645", "data_visualization", "exposure"),

  // ISA 650 — Business Analytics Practicum (graduate)
  L("ISA 650", "consulting", "anchor", "solved a semester-long data-driven business problem for multiple audiences"),
  L("ISA 650", "data_analysis", "applied"),
  L("ISA 650", "presentation_skills", "applied"),
  L("ISA 650", "project_management", "applied"),
  L("ISA 650", "stakeholder_management", "applied"),
  L("ISA 650", "problem_solving", "applied"),
];
