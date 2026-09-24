/**
 * The Job Scout skill taxonomy: one closed vocabulary shared by course
 * mapping, resume extraction, and weekly job tagging, so matching is exact
 * set arithmetic rather than label fuzz (design 2026-07-28).
 *
 * Rules (design §2.1):
 * - An id exists only if employers name the skill in postings AND a student
 *   can evidence it as a unit. Named tools get ids; everything else is an
 *   alias of the id a recruiter would recognize.
 * - `implies` edges are one level deep and point specific → general
 *   (power_bi → data_visualization). Matching grants partial credit across
 *   them in both directions; nothing walks chains.
 * - Aliases are lowercase and exist for deterministic text matching and
 *   highlighting. They never appear in UI copy; labels do.
 *
 * Job tags store TAXONOMY_VERSION so a future vocabulary change can
 * invalidate or migrate old tags instead of silently mismatching.
 */

export const TAXONOMY_VERSION = 2;

export type SkillKind = "tool" | "method" | "domain" | "professional";

export interface SkillDef {
  id: string;
  label: string;
  kind: SkillKind;
  /** Display grouping only; matching never looks at it. */
  category:
    | "programming"
    | "analytics"
    | "machine_learning_ai"
    | "data_management"
    | "visualization_bi"
    | "information_systems"
    | "security_risk"
    | "business"
    | "professional";
  aliases: string[];
  implies: string[];
}

const S = (
  id: string,
  label: string,
  kind: SkillKind,
  category: SkillDef["category"],
  aliases: string[] = [],
  implies: string[] = [],
): SkillDef => ({ id, label, kind, category, aliases, implies });

export const SKILLS: SkillDef[] = [
  // === Programming ===
  S("python", "Python", "tool", "programming", ["python3", "jupyter", "anaconda", "numpy"]),
  S("r", "R", "tool", "programming", ["rstudio", "tidyverse", "r programming", "ggplot2", "shiny"]),
  S("sql", "SQL", "tool", "programming", ["mysql", "postgresql", "t-sql", "pl/sql", "sqlite", "structured query language"]),
  S("excel", "Microsoft Excel", "tool", "programming", ["spreadsheets", "pivot tables", "vlookup", "power query", "vba"]),
  S("sas", "SAS", "tool", "programming", ["sas programming"]),
  S("javascript", "JavaScript", "tool", "programming", ["typescript", "node.js", "js", "es6"]),
  S("pandas", "Pandas", "tool", "programming", ["dataframes"], ["python", "data_wrangling"]),
  S("scikit_learn", "Scikit-learn", "tool", "programming", ["sklearn"], ["python", "machine_learning"]),
  S("deep_learning_frameworks", "Deep Learning Frameworks", "tool", "programming", ["tensorflow", "pytorch", "keras"], ["deep_learning"]),
  S("version_control", "Version Control (Git)", "tool", "programming", ["git", "github", "gitlab", "bitbucket"]),
  S("programming_fundamentals", "Programming Fundamentals", "method", "programming", ["coding", "scripting", "software development"]),
  S("object_oriented_programming", "Object-Oriented Programming", "method", "programming", ["oop", "classes", "inheritance"], ["programming_fundamentals"]),

  // === Analytics methods ===
  S("statistical_analysis", "Statistical Analysis", "method", "analytics", ["statistics", "statistical methods", "descriptive statistics"]),
  S("probability", "Probability", "method", "analytics", ["probability theory", "distributions", "bayesian"]),
  S("statistical_inference", "Statistical Inference", "method", "analytics", ["confidence intervals", "sampling", "estimation"], ["statistical_analysis"]),
  S("hypothesis_testing", "Hypothesis Testing", "method", "analytics", ["t-test", "anova", "chi-square", "significance testing", "p-value"], ["statistical_inference"]),
  S("regression", "Regression Analysis", "method", "analytics", ["linear regression", "logistic regression", "glm", "multiple regression"], ["statistical_analysis"]),
  S("predictive_modeling", "Predictive Modeling", "method", "analytics", ["predictive analytics", "prediction"], []),
  S("forecasting", "Forecasting & Time Series", "method", "analytics", ["time series", "arima", "exponential smoothing", "demand forecasting", "box-jenkins", "time series analysis"], ["predictive_modeling"]),
  S("experimental_design", "Experimental Design", "method", "analytics", ["design of experiments", "doe", "factorial design", "randomized experiments"]),
  S("ab_testing", "A/B Testing", "method", "analytics", ["split testing", "experimentation", "controlled experiments"], ["experimental_design"]),
  S("causal_inference", "Causal Inference", "method", "analytics", ["causal analysis", "difference-in-differences", "counterfactual", "causality"]),
  S("statistical_process_control", "Statistical Process Control", "method", "analytics", ["spc", "control charts", "process monitoring", "quality control"]),
  S("nonparametric_statistics", "Nonparametric Statistics", "method", "analytics", ["rank tests", "distribution-free", "bootstrapping", "resampling"], ["statistical_analysis"]),
  S("optimization", "Optimization", "method", "analytics", ["linear programming", "integer programming", "mathematical programming", "prescriptive analytics", "nonlinear programming"]),
  S("simulation", "Simulation", "method", "analytics", ["monte carlo", "discrete event simulation", "what-if analysis"]),
  S("operations_research", "Operations Research", "domain", "analytics", ["management science", "decision science"]),
  S("network_analysis", "Network Analysis", "method", "analytics", ["graph theory", "social network analysis", "graph analytics"]),
  S("data_analysis", "Data Analysis", "method", "analytics", ["data analytics", "quantitative analysis", "analytical skills"]),
  S("data_wrangling", "Data Wrangling", "method", "analytics", ["data cleaning", "data preparation", "data quality", "data manipulation", "data transformation"]),
  S("data_mining", "Data Mining", "method", "analytics", ["pattern recognition", "knowledge discovery", "market basket analysis"], ["machine_learning"]),
  S("mathematics", "Applied Mathematics", "method", "analytics", ["quantitative skills", "linear algebra", "matrix algebra", "calculus"]),

  // === Machine learning & AI ===
  S("machine_learning", "Machine Learning", "method", "machine_learning_ai", ["ml", "supervised learning", "unsupervised learning"]),
  S("deep_learning", "Deep Learning", "method", "machine_learning_ai", ["neural networks", "cnn", "rnn", "transformers"], ["machine_learning"]),
  S("classification", "Classification Models", "method", "machine_learning_ai", ["decision trees", "random forest", "svm", "naive bayes"], ["machine_learning"]),
  S("clustering", "Clustering & Segmentation", "method", "machine_learning_ai", ["k-means", "hierarchical clustering", "segmentation", "dbscan"], ["machine_learning"]),
  S("ensemble_methods", "Ensemble Methods", "method", "machine_learning_ai", ["xgboost", "gradient boosting", "bagging", "boosting"], ["machine_learning"]),
  S("model_evaluation", "Model Evaluation", "method", "machine_learning_ai", ["cross-validation", "model validation", "precision", "recall", "auc"], ["machine_learning"]),
  S("nlp", "Natural Language Processing", "method", "machine_learning_ai", ["text mining", "text analytics", "sentiment analysis", "natural language processing"], ["machine_learning"]),
  S("computer_vision", "Computer Vision", "method", "machine_learning_ai", ["image processing", "image recognition"], ["deep_learning"]),
  S("anomaly_detection", "Anomaly Detection", "method", "machine_learning_ai", ["outlier detection", "fraud detection"], ["machine_learning"]),
  S("generative_ai", "Generative AI", "method", "machine_learning_ai", ["gen ai", "llm", "large language models", "chatgpt", "foundation models"]),
  S("prompt_engineering", "Prompt Engineering", "method", "machine_learning_ai", ["prompting", "prompt design"], ["generative_ai"]),
  S("llm_applications", "LLM Applications", "method", "machine_learning_ai", ["rag", "retrieval augmented generation", "ai agents", "openai api", "ai integration"], ["generative_ai"]),
  S("ai_ethics", "Responsible AI", "domain", "machine_learning_ai", ["ai governance", "ai bias", "ethical ai", "responsible ai"]),

  // === Data management & engineering ===
  S("database_design", "Database Design", "method", "data_management", ["schema design", "normalization", "er modeling", "entity relationship", "relational databases"]),
  S("data_warehousing", "Data Warehousing", "method", "data_management", ["olap", "star schema", "dimensional modeling", "data marts", "data warehouse"]),
  S("etl", "ETL & Data Pipelines", "method", "data_management", ["elt", "data pipelines", "data integration", "ssis", "airflow", "dbt"]),
  S("nosql", "NoSQL Databases", "tool", "data_management", ["mongodb", "cassandra", "document databases", "redis"]),
  S("big_data", "Big Data", "method", "data_management", ["distributed computing", "large-scale data"]),
  S("spark", "Apache Spark", "tool", "data_management", ["pyspark", "spark sql"], ["big_data"]),
  S("databricks", "Databricks", "tool", "data_management", ["delta lake", "lakehouse"], ["big_data"]),
  S("hadoop", "Hadoop", "tool", "data_management", ["hdfs", "mapreduce", "hive"], ["big_data"]),
  S("streaming_analytics", "Streaming Analytics", "method", "data_management", ["real-time analytics", "kafka", "stream processing"], ["big_data"]),
  S("data_lakes", "Data Lakes", "method", "data_management", ["raw data storage"], ["data_warehousing"]),
  S("cloud_computing", "Cloud Computing", "method", "data_management", ["iaas", "paas", "saas", "cloud platforms"]),
  S("aws", "Amazon Web Services", "tool", "data_management", ["amazon web services", "s3", "redshift", "ec2"], ["cloud_computing"]),
  S("azure", "Microsoft Azure", "tool", "data_management", ["synapse", "azure data factory", "microsoft fabric"], ["cloud_computing"]),
  S("gcp", "Google Cloud Platform", "tool", "data_management", ["google cloud", "bigquery"], ["cloud_computing"]),
  S("snowflake", "Snowflake", "tool", "data_management", ["snowsql"], ["data_warehousing", "cloud_computing"]),
  S("data_engineering", "Data Engineering", "domain", "data_management", ["data infrastructure", "dataops"]),
  S("data_architecture", "Data Architecture", "method", "data_management", ["enterprise data architecture", "data modeling"]),
  S("data_governance", "Data Governance", "method", "data_management", ["master data management", "data stewardship", "metadata management", "data quality management"]),
  S("data_privacy", "Data Privacy", "method", "data_management", ["gdpr", "ccpa", "privacy compliance", "data protection"]),

  // === Visualization & BI ===
  S("data_visualization", "Data Visualization", "method", "visualization_bi", ["data viz", "charts", "visual analytics"]),
  S("tableau", "Tableau", "tool", "visualization_bi", ["tableau desktop", "tableau server", "tableau prep"], ["data_visualization"]),
  S("power_bi", "Power BI", "tool", "visualization_bi", ["powerbi", "dax"], ["data_visualization"]),
  S("dashboard_design", "Dashboard Design", "method", "visualization_bi", ["kpi dashboards", "executive dashboards", "interactive dashboards"], ["data_visualization"]),
  S("business_intelligence", "Business Intelligence", "domain", "visualization_bi", ["bi", "reporting", "bi tools"]),
  S("data_storytelling", "Data Storytelling", "professional", "visualization_bi", ["insight communication", "narrative analytics"], ["communication"]),

  // === Information systems & business ===
  S("information_systems", "Information Systems", "domain", "information_systems", ["mis", "business systems", "is"]),
  S("enterprise_systems", "Enterprise Systems (ERP)", "domain", "information_systems", ["erp", "sap", "oracle erp", "enterprise resource planning"]),
  S("crm", "CRM Systems", "tool", "information_systems", ["salesforce", "customer relationship management"]),
  S("supply_chain", "Supply Chain Management", "domain", "information_systems", ["scm", "logistics", "inventory management"]),
  S("systems_analysis", "Systems Analysis", "method", "information_systems", ["business analysis", "requirements gathering", "requirements analysis", "business requirements"]),
  S("sdlc", "Systems Development Life Cycle", "method", "information_systems", ["software development life cycle", "systems development", "software acquisition"]),
  S("agile", "Agile Methods", "method", "information_systems", ["scrum", "kanban", "sprint planning", "jira"]),
  S("project_management", "Project Management", "method", "information_systems", ["pmp", "project planning", "project delivery"]),
  S("business_process", "Business Process Analysis", "method", "information_systems", ["process improvement", "process mapping", "bpmn", "business process management"]),
  S("it_strategy", "IT Strategy", "domain", "information_systems", ["digital strategy", "technology strategy", "technology planning"]),
  S("digital_transformation", "Digital Transformation", "domain", "information_systems", ["digitization", "digital innovation", "emerging technologies"]),
  S("it_management", "IT Management", "domain", "information_systems", ["technology management", "it leadership", "it operations"]),
  S("web_development", "Web Development", "method", "information_systems", ["frontend", "backend", "full stack", "html", "css", "react"], ["programming_fundamentals"]),
  S("mobile_development", "Mobile Development", "method", "information_systems", ["ios", "android", "react native", "mobile apps"], ["programming_fundamentals"]),
  S("api_development", "APIs & Integration", "method", "information_systems", ["rest api", "web services", "microservices", "api integration"]),
  S("networking", "Networking & Data Communications", "method", "information_systems", ["tcp/ip", "lan", "wan", "network administration", "data communications", "wireless networks"]),
  S("blockchain", "Blockchain", "method", "information_systems", ["distributed ledger", "smart contracts", "cryptocurrency", "web3", "defi"]),
  S("marketing_analytics", "Marketing & Customer Analytics", "domain", "information_systems", ["customer analytics", "churn analysis", "campaign analysis"]),

  // === Security & risk ===
  S("cybersecurity", "Cybersecurity", "domain", "security_risk", ["information security", "infosec", "cyber defense"]),
  S("security_operations", "Security Operations & Analytics", "method", "security_risk", ["siem", "soc", "incident response", "threat detection", "security monitoring", "intrusion detection", "security analytics", "threat intelligence"], ["cybersecurity"]),
  S("vulnerability_management", "Vulnerability Management", "method", "security_risk", ["penetration testing", "vulnerability assessment", "ethical hacking"], ["cybersecurity"]),
  S("encryption", "Encryption & Cryptography", "method", "security_risk", ["cryptography", "pki", "tls", "hashing"], ["cybersecurity"]),
  S("identity_access_management", "Identity & Access Management", "method", "security_risk", ["iam", "access control", "authentication", "authorization"], ["cybersecurity"]),
  S("risk_management", "Risk Management", "method", "security_risk", ["risk assessment", "enterprise risk", "risk analysis"]),
  S("it_audit", "IT Audit", "method", "security_risk", ["security audit", "it controls", "sox testing", "cisa", "audit analytics"]),
  S("compliance", "Compliance", "method", "security_risk", ["regulatory compliance", "sox", "hipaa", "pci dss"]),
  S("it_governance", "IT Governance", "method", "security_risk", ["cobit", "itil", "security policy", "governance frameworks"]),
  S("business_continuity", "Business Continuity", "method", "security_risk", ["disaster recovery", "incident planning", "resilience"]),

  // === Business domains (v2, 2026-09-24) ===
  // Added with the FSB course catalog so finance, accounting, marketing,
  // management, and economics courses have skills to map to. Kept at the
  // level analytics postings ask for ("finance background preferred"), and
  // designed to split later through `implies` edges when the job feed grows
  // (professor's decision). The tagger tags these only when the role's
  // duties or qualifications ask for them, never from the employer's
  // industry, and defaults them to "preferred".
  S("financial_accounting", "Financial Accounting & Reporting", "domain", "business", ["financial reporting", "gaap", "ifrs", "financial statements", "accounting"]),
  S("managerial_accounting", "Managerial & Cost Accounting", "domain", "business", ["cost accounting", "management accounting", "budgeting", "variance analysis", "budget formulation", "budget execution", "cost analysis"]),
  S("auditing", "Auditing", "domain", "business", ["financial audit", "external audit", "internal audit", "assurance"]),
  S("tax", "Tax", "domain", "business", ["taxation", "tax accounting", "income tax", "tax planning"]),
  S("corporate_finance", "Corporate Finance", "domain", "business", ["capital budgeting", "cost of capital", "capital structure", "financial management", "working capital"]),
  S("financial_analysis", "Financial Analysis", "method", "business", ["financial statement analysis", "ratio analysis", "fp&a", "financial planning and analysis"]),
  S("financial_modeling", "Financial Modeling & Valuation", "method", "business", ["valuation", "dcf", "discounted cash flow", "company valuation", "financial model"], ["financial_analysis"]),
  S("investments", "Investments & Portfolio Management", "domain", "business", ["portfolio management", "asset management", "equity research", "securities analysis", "fixed income", "derivatives"]),
  S("financial_planning", "Personal Financial Planning", "domain", "business", ["personal finance", "wealth management", "retirement planning", "financial advising"]),
  S("real_estate", "Real Estate Analysis", "domain", "business", ["real estate finance", "property valuation", "commercial real estate", "real estate development"]),
  S("economics", "Economic Analysis", "domain", "business", ["microeconomics", "macroeconomics", "economic modeling", "economic research"]),
  S("econometrics", "Econometrics", "method", "business", ["panel data", "instrumental variables", "applied econometrics"], ["regression"]),
  S("marketing_strategy", "Marketing Strategy", "domain", "business", ["marketing management", "go-to-market", "market positioning", "marketing planning", "segmentation and targeting"]),
  S("market_research", "Market Research", "method", "business", ["marketing research", "consumer research", "survey design", "focus groups"]),
  S("consumer_behavior", "Consumer Behavior", "domain", "business", ["buyer behavior", "consumer psychology", "shopper insights"]),
  S("digital_marketing", "Digital Marketing", "domain", "business", ["seo", "search engine optimization", "social media marketing", "content marketing", "email marketing", "sem"]),
  S("sales", "Sales", "professional", "business", ["professional selling", "business development", "account management", "sales management"]),
  S("brand_management", "Brand Management", "domain", "business", ["branding", "brand strategy", "advertising", "integrated marketing communications"]),
  S("operations_management", "Operations Management", "domain", "business", ["production planning", "capacity planning", "operations planning", "service operations"]),
  S("process_improvement", "Process Improvement (Lean and Six Sigma)", "method", "business", ["lean", "six sigma", "lean six sigma", "continuous improvement", "kaizen"]),
  S("human_capital_management", "Human Capital Management", "domain", "business", ["human resources", "hr", "hrm", "talent management", "recruiting", "compensation and benefits", "workforce planning"]),
  S("organizational_behavior", "Organizational Behavior", "domain", "business", ["organizational psychology", "team dynamics", "change management", "organizational culture"]),
  S("negotiation", "Negotiation", "professional", "business", ["negotiating", "conflict resolution", "deal making"]),
  S("entrepreneurship", "Entrepreneurship", "domain", "business", ["new ventures", "startups", "venture creation", "business planning", "business model canvas"]),
  S("design_thinking", "Innovation & Design Thinking", "method", "business", ["design thinking", "human-centered design", "innovation management", "creative problem solving"]),
  S("business_law", "Business Law", "domain", "business", ["contract law", "legal environment of business", "employment law", "corporate law"]),
  S("international_business", "International Business", "domain", "business", ["global business", "international trade", "cross-cultural management", "global markets"]),
  S("sustainability", "Sustainability", "domain", "business", ["esg", "corporate social responsibility", "csr", "sustainable business"]),
  S("business_ethics", "Business Ethics", "domain", "business", ["corporate ethics", "ethical decision making", "corporate governance"]),
  S("strategic_management", "Strategic Management", "domain", "business", ["competitive strategy", "corporate strategy", "strategy execution"], ["business_strategy"]),

  // === Professional ===
  S("communication", "Communication", "professional", "professional", ["written communication", "verbal communication", "interpersonal skills"]),
  S("presentation_skills", "Presentation Skills", "professional", "professional", ["public speaking", "powerpoint", "presentations"], ["communication"]),
  S("technical_writing", "Technical Writing", "professional", "professional", ["documentation", "white papers", "technical reports", "reproducible reporting"], ["communication"]),
  S("consulting", "Consulting", "professional", "professional", ["advisory", "client engagement", "client-facing"]),
  S("stakeholder_management", "Stakeholder Management", "professional", "professional", ["client communication", "stakeholder engagement", "cross-functional collaboration"]),
  S("problem_solving", "Problem Solving", "professional", "professional", ["critical thinking", "analytical thinking", "troubleshooting"]),
  S("teamwork", "Teamwork", "professional", "professional", ["collaboration", "team projects"]),
  S("leadership", "Leadership", "professional", "professional", ["mentoring", "team leadership"]),
  S("business_strategy", "Business Strategy", "domain", "professional", ["strategic planning", "competitive analysis"]),
  S("business_acumen", "Business Acumen", "professional", "professional", ["business knowledge", "domain knowledge", "commercial awareness"]),
  S("research", "Research", "professional", "professional", ["research methods", "literature review", "independent research"]),
  S("ethics", "Professional Ethics", "professional", "professional", ["information ethics", "data ethics"]),
];

export const SKILL_IDS = SKILLS.map((s) => s.id);

const byId = new Map(SKILLS.map((s) => [s.id, s]));

export function getSkill(id: string): SkillDef | undefined {
  return byId.get(id);
}

const byAlias = new Map<string, string>();
for (const s of SKILLS) {
  byAlias.set(s.label.toLowerCase(), s.id);
  for (const a of s.aliases) byAlias.set(a, s.id);
}

/**
 * Where a text names a skill by its label or an alias, as whole words
 * ("hr" never matches inside "three"). Returns the matched snippet with some
 * context, or null. Used to require textual evidence for business-domain
 * job tags (v6.7.0), so a tag reflects what the posting asks for rather than
 * a guess from the employer's industry.
 */
export function mentionsSkill(skillId: string, text: string): string | null {
  const skill = byId.get(skillId);
  if (!skill) return null;
  for (const term of [skill.label.toLowerCase(), ...skill.aliases]) {
    const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const m = new RegExp(`(?<![a-z0-9])${escaped}s?(?![a-z0-9])`, "i").exec(text);
    if (m) return text.slice(Math.max(0, m.index - 60), m.index + m[0].length + 60).replace(/\s+/g, " ").trim();
  }
  return null;
}

/**
 * Maps a model-emitted skill string onto the closed vocabulary, or null.
 * Exists because Gemini rejects a 104-value enum in its response schema
 * (found live, 2026-07-28), so the wire schema accepts plain strings and
 * THIS is the enforcement: exact id, then label/alias, then simple
 * space-to-underscore normalization. Anything unresolved is dropped by the
 * caller; the vocabulary never grows from model output.
 */
export function resolveSkillId(raw: string): string | null {
  const cleaned = raw.trim().toLowerCase();
  if (byId.has(cleaned)) return cleaned;
  const aliased = byAlias.get(cleaned);
  if (aliased) return aliased;
  const underscored = cleaned.replace(/[\s-]+/g, "_");
  return byId.has(underscored) ? underscored : null;
}
