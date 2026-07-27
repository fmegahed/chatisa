/**
 * The ISA course catalog. "Course" here is only a label a student selects for
 * their own project. The app looks up no enrollment data and stores no roster.
 * Codes are combined for dual-listed courses (e.g. "401/501").
 */
export interface Course {
  code: string;
  title: string;
}

export const ISA_COURSES: readonly Course[] = [
  { code: "125", title: "Introduction to Business Statistics" },
  { code: "177", title: "Independent Studies" },
  { code: "211", title: "Information Technology and Data Driven Decision Making in Business" },
  { code: "225", title: "Principles of Business Analytics" },
  { code: "235", title: "Information Technology and the Intelligent Enterprise" },
  { code: "241", title: "Database for Analytics" },
  { code: "242", title: "Programming for Analytics" },
  { code: "250", title: "Basic Math for Analytics" },
  { code: "277", title: "Independent Studies" },
  { code: "301", title: "Business Data Communications and Security" },
  { code: "303", title: "Enterprise Systems" },
  { code: "305", title: "Information Technology Governance, Risk Management, Security and Audit" },
  { code: "321", title: "Optimization in Business Analytics" },
  { code: "333", title: "Nonparametric Statistics" },
  { code: "335", title: "Blockchain and Business Applications" },
  { code: "336", title: "Generative AI in Business" },
  { code: "340", title: "Internship" },
  { code: "345", title: "Database Systems and Data Warehousing" },
  { code: "365", title: "Statistical Monitoring and Design of Experiments" },
  { code: "377", title: "Independent Studies" },
  { code: "381", title: "Concepts in Business Programming" },
  { code: "387", title: "Designing Business Systems" },
  { code: "391", title: "Applied Regression Analysis in Business" },
  { code: "401/501", title: "Business Intelligence and Data Visualization" },
  { code: "403", title: "Building Web and Mobile Business Applications" },
  { code: "405", title: "Information Security" },
  { code: "406", title: "IT Project Management" },
  { code: "414/514", title: "Managing Big Data" },
  { code: "419", title: "Data Driven Security" },
  { code: "424", title: "Data Infrastructure for the Enterprise" },
  { code: "444/544", title: "Business Forecasting" },
  { code: "477", title: "Independent Studies" },
  { code: "480", title: "Topics in Business Analytics" },
  { code: "481", title: "Topics in Information Systems" },
  { code: "491/591", title: "Introduction to Data Mining in Business" },
  { code: "495", title: "Managing the Intelligent Enterprise" },
  { code: "496", title: "Business Analytics Practicum" },
  { code: "612", title: "Advanced Business Intelligence" },
  { code: "616", title: "Communicating with Data" },
  { code: "621", title: "Enabling Technology Topics I" },
  { code: "628", title: "Information Technology and Analytic's Role in the Enterprise" },
  { code: "629", title: "Leveraging IT and Data Across the Business" },
  { code: "630", title: "Machine Learning Applications in Business" },
  { code: "632", title: "Big Data Analytics and Modern AI" },
  { code: "633", title: "Experimental Design and Causal Methods" },
  { code: "634", title: "Systems Modeling and Optimization" },
  { code: "641", title: "Data Discovery Through Business Analytics for Managers" },
  { code: "645", title: "Business Analytics for the Executive" },
  { code: "650", title: "Business Analytics Practicum" },
  { code: "677", title: "Independent Studies" },
];

const BY_CODE = new Map(ISA_COURSES.map((c) => [c.code, c]));

export function findCourse(code: string): Course | undefined {
  return BY_CODE.get(code);
}

/** "ISA 401/501: Business Intelligence and Data Visualization". No em dash. */
export function courseLabel(course: Course): string {
  return `ISA ${course.code}: ${course.title}`;
}
