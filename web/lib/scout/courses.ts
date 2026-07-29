/**
 * The ISA course catalog Job Scout matches against. Snapshot of
 * bulletin.miamioh.edu/courses-instruction/isa/ fetched 2026-07-28 (user
 * decision: the live bulletin replaces the stale careerbridge JSON, and
 * Independent Studies 177/277/377/477/677 are excluded).
 *
 * `special: "freeform"` courses (Internship, Topics seminars) have no static
 * skill mapping: their content varies per student, so the profile asks one
 * line about what they worked on and maps it with a model instead.
 *
 * Descriptions are trimmed to what a skill reviewer needs; the bulletin is
 * the source of truth for full text. Credits drive match weighting: a
 * 1.5-credit course contributes half the depth of a 3-credit one.
 */

export interface CourseDef {
  /** Primary code as students know it, e.g. "ISA 401". */
  code: string;
  /** Cross-listed codes that count as the same course (STA/ACC/BUS/5xx). */
  altCodes: string[];
  title: string;
  credits: number;
  description: string;
  special?: "freeform";
}

const C = (
  code: string,
  altCodes: string[],
  title: string,
  credits: number,
  description: string,
  special?: "freeform",
): CourseDef => ({ code, altCodes, title, credits, description, special });

export const COURSES: CourseDef[] = [
  C("ISA 125", ["STA 125"], "Introduction to Business Statistics", 3,
    "Data, probability, sampling, and analytical decision-making; summarizing data, relationships among variables, one- and two-sample inference."),
  C("ISA 211", [], "Information Technology and Data Driven Decision Making in Business", 3,
    "Information systems and analytics for the non-business major; how organizations use IT and analytics for data-driven decisions."),
  C("ISA 225", [], "Principles of Business Analytics", 3,
    "Probability and classification, data visualization, inference, predictive modeling with regression, forecasting, and data mining, with computer implementation on real data."),
  C("ISA 235", [], "Information Technology and the Intelligent Enterprise", 3,
    "Strategic and transformational role of IT and data; technology-driven innovation, information ethics, data management, problem-solving with spreadsheets, visualization, and AI."),
  C("ISA 241", [], "Database for Analytics", 1.5,
    "Collection, manipulation, and management of structured data; logical and physical database design, entity relationship modeling, and SQL."),
  C("ISA 242", [], "Programming for Analytics", 1.5,
    "Programming skills to access and process data; structured techniques and libraries for data retrieval, logic, and presentation."),
  C("ISA 250", ["STA 250"], "Basic Math for Analytics", 3,
    "Applied foundations: sets, functions, logarithms, exponentials, matrix algebra, introductory calculus, and basic optimization, software-driven."),
  C("ISA 301", [], "Business Data Communications and Security", 3,
    "Data communications in business: network architectures, wired/wireless standards, network and data security, protective technologies, cloud computing."),
  C("ISA 303", [], "Enterprise Systems", 3,
    "ERP, supply chain, and CRM systems; managerial and technological considerations in implementation and use."),
  C("ISA 305", ["ACC 305"], "Information Technology Governance, Risk Management, Security and Audit", 3,
    "Foundations of IT risk management, security, and assurance; managerial strategy and technical controls."),
  C("ISA 321", [], "Optimization in Business Analytics", 3,
    "Prescriptive models: linear, integer, and nonlinear programming and network analytics for production, supply chain, labor, finance, and social networks."),
  C("ISA 333", ["STA 333"], "Nonparametric Statistics", 3,
    "Statistical techniques when the underlying distribution is unknown: chi-square, runs, and association tests."),
  C("ISA 335", [], "Blockchain and Business Applications", 3,
    "Blockchain components and cryptographic techniques; applications including cryptocurrencies, smart contracts, tokens, DAOs, and DeFi."),
  C("ISA 336", [], "Generative AI in Business", 3,
    "Large language models, prompt engineering, and multimodality; applications for content creation and process automation, integration with external data, and societal and ethical issues."),
  C("ISA 340", [], "Internship", 1,
    "Supervised business internship with a faculty sponsor and reflection paper.", "freeform"),
  C("ISA 345", [], "Database Systems and Data Warehousing", 3,
    "Database concepts, design methodologies, DBMS, SQL, implementation, and data warehousing."),
  C("ISA 365", ["STA 365"], "Statistical Monitoring and Design of Experiments", 3,
    "Statistical methods for monitoring process data and data streams; experimental design applied to business analytics."),
  C("ISA 381", [], "Concepts in Business Programming", 3,
    "Advanced structuring, design, and development of scalable data-driven applications; structured and object-oriented programming, functions, lambda expressions, and custom libraries across TXT, CSV, Excel, and databases."),
  C("ISA 387", [], "Designing Business Systems", 3,
    "Planning, evaluating, and acquiring business software: development, outsourcing, purchase; application life cycle, methods, techniques, and tools."),
  C("ISA 391", [], "Applied Regression Analysis in Business", 3,
    "Multiple regression for business problems, explanatory and predictive; inference, assumptions, model building, and evaluation."),
  C("ISA 401", ["ISA 501"], "Business Intelligence and Data Visualization", 3,
    "Business intelligence and data visualization in organizations: how information is gathered, stored, analyzed, and used; data warehousing and data mining."),
  C("ISA 403", [], "Building Web and Mobile Business Applications", 3,
    "Design and development of scalable web and web-based mobile applications using client and server-side technologies."),
  C("ISA 405", [], "Information Security", 3,
    "Threats, vulnerabilities, encryption, controls, privacy; governance, policy, risk frameworks, business continuity, compliance, and ethics, with case studies and security tools."),
  C("ISA 406", [], "IT Project Management", 3,
    "IT project management theories, techniques, and software tools, focused on modern IT and software implementation projects."),
  C("ISA 414", ["ISA 514"], "Managing Big Data", 3,
    "Theories and technologies for extracting insight from unstructured and large-scale data; big data solutions for business decisions."),
  C("ISA 419", [], "Data Driven Security", 3,
    "Data-driven security analytics: malicious pattern discovery in security logs, user behavior analysis, intrusion detection, web security, phishing detection, and IIoT security, with extensive programming on real datasets."),
  C("ISA 424", [], "Data Infrastructure for the Enterprise", 3,
    "Data infrastructure for decision making: data warehouses, data lakes, NoSQL systems, and cloud computing, plus managerial issues."),
  C("ISA 444", ["ISA 544"], "Business Forecasting", 3,
    "Analyzing and forecasting business time series: Box-Jenkins, time series regression with autocorrelated errors, exponential smoothing, and classical decomposition."),
  C("ISA 480", [], "Topics in Business Analytics", 3,
    "Seminar on significant emerging topics in business analytics.", "freeform"),
  C("ISA 481", [], "Topics in Information Systems", 3,
    "Seminar on significant emerging topics in information systems.", "freeform"),
  C("ISA 491", ["ISA 591"], "Introduction to Data Mining in Business", 3,
    "Analysis of large business datasets: cluster analysis, market basket analysis, trees, logistic regression, neural nets, and model evaluation with current software."),
  C("ISA 495", [], "Managing the Intelligent Enterprise", 3,
    "Independent research on a topic and company from an MIS perspective; analytical and creative case responses presented to the class."),
  C("ISA 496", [], "Business Analytics Practicum", 3,
    "Analytics consulting for real business clients using data mining, visualization, modeling, and data skills from previous courses."),
  C("ISA 612", [], "Advanced Business Intelligence", 3,
    "Business intelligence and its data infrastructure: retrieving, cleaning, manipulating, and modeling structured data; interactive dashboards with charts, filters, and KPIs."),
  C("ISA 616", [], "Communicating with Data", 3,
    "From client consultation to implementation to presentation: communicating quantitative analyses, visualization, reproducible documentation, professional white papers, and ethics."),
  C("ISA 621", [], "Enabling Technology Topics I", 3,
    "Existing and emerging IT in the organization: IT's role in business processes, innovation methodology, and infrastructure technologies."),
  C("ISA 628", [], "Information Technology and Analytic's Role in the Enterprise", 1.5,
    "Existing and emerging IT for reinventing processes and consuming data to improve decisions; IT's role in business processes and data leverage."),
  C("ISA 629", [], "Leveraging IT and Data Across the Business", 1.5,
    "Common technologies and techniques for data manipulation and consumption across business processes; applied to discipline-specific problems."),
  C("ISA 630", [], "Machine Learning Applications in Business", 3,
    "Supervised and unsupervised modeling with AI and machine learning: ensembles, customized ensembles, and deep learning, with business impact focus."),
  C("ISA 632", [], "Big Data Analytics and Modern AI", 3,
    "In-memory cluster computing, non-relational storage, data lakes, data governance; distributed machine learning, streaming analytics, large-scale network analysis, NLP, speech, and image processing."),
  C("ISA 633", [], "Experimental Design and Causal Methods", 3,
    "Discovering causal relationships in business: A/B testing, complex experiments, and causal inference modeling when experimentation is infeasible."),
  C("ISA 634", [], "Systems Modeling and Optimization", 3,
    "Designing and optimizing complex systems: mathematical programming under business metrics and constraints, graph-theoretic network modeling, prescriptive analytics."),
  C("ISA 641", [], "Data Discovery Through Business Analytics for Managers", 2,
    "Basic tools and methods of data-driven decision making, with an open-source programming introduction applied to summarization, visualization, and discovery."),
  C("ISA 645", ["BUS 645"], "Business Analytics for the Executive", 3,
    "Analysis measures and methods leading organizations employ for data-driven business and marketing decisions."),
  C("ISA 650", [], "Business Analytics Practicum", 3,
    "Immersive project-based practicum: semester-long data-driven problem solving with findings communicated to multiple audiences."),
];

export function getCourse(code: string): CourseDef | undefined {
  return COURSES.find(
    (c) => c.code === code || c.altCodes.includes(code),
  );
}

/**
 * The courses shown before "Show more" in each tier of the profile picker
 * (instructor's own popularity call, 2026-07-29). Everything else in the
 * tier sits behind a disclosure; the graduate tier is entirely collapsed.
 * A unit test asserts every code here exists in COURSES.
 */
export const POPULAR_CODES: Record<string, string[]> = {
  foundations: ["ISA 125", "ISA 225", "ISA 235"],
  core300: [
    "ISA 301", "ISA 303", "ISA 305", "ISA 321", "ISA 336",
    "ISA 345", "ISA 365", "ISA 381", "ISA 387", "ISA 391",
  ],
  advanced400: [
    "ISA 401", "ISA 403", "ISA 405", "ISA 406", "ISA 414",
    "ISA 419", "ISA 444", "ISA 491", "ISA 495", "ISA 496",
  ],
};
