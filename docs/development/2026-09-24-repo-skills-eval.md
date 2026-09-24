# Repository skills: which model (v6.8.0 decision)

**Decision: GPT-6 Luna is the default** for suggesting skills from repositories. It passes all three rules agreed in advance with the professor:
- **(a) failures:** Luna failed on 0 of the 8 repositories; the rule allows at most 1.
- **(b) coverage:** Luna suggests 76% of the applied-or-anchor skills Sol suggests (81% on an earlier run); the bar is 75%.
- **(c) plainly wrong suggestions:** none (checks below).

It costs about 2% as much as GPT-6 Sol: $0.02 for all eight repositories against $0.43 for Sol. Students can still pick another model in the block's chooser.

**Setup:**
- Repositories: 8 of the professor's public repositories, named by him on 2026-09-24, read-only with his GitHub CLI token on this machine.
- Code: the final branch code (the summary reader, the guards and `suggestRepoSkills`), so these are exactly the suggestions a student would see before confirming.

## Checking suggestions against the code

Every doubtful suggestion was checked in the repository itself:

- **Luna, Pandas in `llm_consistency`** (an R and HTML repository): correct. `rmarkdown/llm_consistency.Rmd` has Python chunks with `import pandas as pd`. Sol missed it.
- **Web Development in `tavr_paper`** (suggested by Luna, and by Sol and Gemini on other runs): correct. `notebooks/deployed_model.ipynb` launches a Gradio interface (`gr.Interface(...)`), and the requirements include Dash, Flask and FastAPI. I first judged this wrong from the file list alone; reading the notebook showed otherwise.
- **Luna, Classification Models in `llm_consistency`:** correct. The project runs binary classification experiments, with LLMs labelling news sentiment.
- **Sol, Classification Models in `covid19-deaths`:** wrong, on both runs. The repository does time-series clustering; "classification" appears only as a heading about a derived region variable ("State Region Classification").

## Fallback checked: Gemini 3.8 Flash

Gemini 3.8 Flash is the job tagger's model and was the agreed fallback.
- It failed on 2 of the 8 repositories (no usable structured answer for `chatsqc` and `vaccines_spatial_and_optimization`), which fails rule (a).
- Its coverage of Sol was 86%.

## A fix found by the eval

`renv/activate.R` (generated bootstrap code) was taking one of the six code-file slots in R repositories. The reader now treats the whole `renv/` folder as generated, with a test. The runs below use the fixed reader.

## Final run: GPT-6 Luna vs GPT-6 Sol

Repositories: 8. Failures: gpt-6-luna 0, gpt-6-sol 0. gpt-6-luna covers 45 of gpt-6-sol's 59 applied-or-anchor skills (76%).

| Repository | Model | Anchors | Applied | Exposure | Failed | Cost |
|---|---|---|---|---|---|---|
| fmegahed/chatsqc | gpt-6-luna |  | Python, LLM Applications, Prompt Engineering, Web Development, Data Engineering, Data Wrangling, Generative AI |  | | $0.0021 |
| fmegahed/chatsqc | gpt-6-sol |  | Python, LLM Applications, Prompt Engineering, Natural Language Processing, ETL & Data Pipelines, Data Wrangling, Web Development, Pandas |  | | $0.0415 |
| fmegahed/llm_expository | gpt-6-luna |  | R, Statistical Process Control, Data Visualization, Data Wrangling |  | | $0.0009 |
| fmegahed/llm_expository | gpt-6-sol |  | Statistical Process Control, Generative AI, R, Data Wrangling, Data Visualization |  | | $0.0165 |
| fmegahed/conformal_clip | gpt-6-luna | Machine Learning, Classification Models, Computer Vision | Python, Statistical Analysis, Scikit-learn, Pandas, Data Visualization |  | | $0.0030 |
| fmegahed/conformal_clip | gpt-6-sol | Computer Vision, Statistical Inference, Classification Models | Model Evaluation, Python, Scikit-learn, Pandas |  | | $0.0585 |
| fmegahed/llm_consistency | gpt-6-luna |  | R, Statistical Analysis, Classification Models, LLM Applications, Pandas, Prompt Engineering, Data Visualization |  | | $0.0050 |
| fmegahed/llm_consistency | gpt-6-sol |  | LLM Applications, Statistical Analysis, Model Evaluation, Prompt Engineering, R, Experimental Design, Data Visualization |  | | $0.0963 |
| fmegahed/shoulder_fatigue_modeling | gpt-6-luna |  | Python, Pandas, Predictive Modeling, Regression Analysis, Machine Learning, Model Evaluation, Data Analysis, Data Visualization |  | | $0.0021 |
| fmegahed/shoulder_fatigue_modeling | gpt-6-sol |  | Python, Pandas, Machine Learning, Regression Analysis, Classification Models, Model Evaluation, Data Analysis, Data Visualization |  | | $0.0407 |
| fmegahed/tavr_paper | gpt-6-luna |  | Python, Predictive Modeling, Classification Models, Pandas, Data Wrangling, Model Evaluation, Scikit-learn, Web Development |  | | $0.0016 |
| fmegahed/tavr_paper | gpt-6-sol |  | Predictive Modeling, Classification Models, Regression Analysis, Model Evaluation, Python, Pandas, Data Wrangling, Web Development |  | | $0.0318 |
| fmegahed/vaccines_spatial_and_optimization | gpt-6-luna | R, Statistical Analysis, Regression Analysis | Predictive Modeling, Data Wrangling, Data Visualization | Optimization | | $0.0051 |
| fmegahed/vaccines_spatial_and_optimization | gpt-6-sol | R, Data Wrangling, Regression Analysis | Predictive Modeling, Statistical Inference, Model Evaluation, Data Visualization, Data Analysis |  | | $0.0995 |
| fmegahed/covid19-deaths | gpt-6-luna | R, Data Analysis, Clustering & Segmentation | Statistical Analysis, Regression Analysis, Data Wrangling, Data Visualization, Model Evaluation |  | | $0.0024 |
| fmegahed/covid19-deaths | gpt-6-sol | R, Clustering & Segmentation, Regression Analysis | Data Wrangling, Data Visualization, Model Evaluation, Classification Models, Statistical Analysis |  | | $0.0463 |

## Fallback run: Gemini 3.8 Flash vs GPT-6 Sol

Repositories: 8. Failures: gemini-3.8-flash 2, gpt-6-sol 0. gemini-3.8-flash covers 37 of gpt-6-sol's 43 applied-or-anchor skills (86%).

| Repository | Model | Anchors | Applied | Exposure | Failed | Cost |
|---|---|---|---|---|---|---|
| fmegahed/chatsqc | gemini-3.8-flash | | | | AI_NoObjectGeneratedError: No object generated: could not parse the response. | |
| fmegahed/chatsqc | gpt-6-sol |  | Python, LLM Applications, Prompt Engineering, Generative AI, Natural Language Processing, ETL & Data Pipelines, Data Wrangling, Web Development |  | | $0.0410 |
| fmegahed/llm_expository | gemini-3.8-flash |  | Statistical Process Control, R, Data Visualization, Data Wrangling, Generative AI, Prompt Engineering |  | | $0.0107 |
| fmegahed/llm_expository | gpt-6-sol |  | Statistical Process Control, R, Data Visualization, Data Wrangling, Generative AI |  | | $0.0165 |
| fmegahed/conformal_clip | gemini-3.8-flash | Python, Computer Vision, Machine Learning | Classification Models, Deep Learning, Scikit-learn, Model Evaluation |  | | $0.0289 |
| fmegahed/conformal_clip | gpt-6-sol | Python, Computer Vision, Classification Models | Statistical Inference, Model Evaluation, Scikit-learn, Pandas |  | | $0.0584 |
| fmegahed/llm_consistency | gemini-3.8-flash |  | R, LLM Applications, Statistical Analysis, Model Evaluation, Prompt Engineering, Data Visualization, Natural Language Processing |  | | $0.0456 |
| fmegahed/llm_consistency | gpt-6-sol |  | R, LLM Applications, Prompt Engineering, Statistical Inference, Model Evaluation, Classification Models, Data Visualization |  | | $0.0964 |
| fmegahed/shoulder_fatigue_modeling | gemini-3.8-flash |  | Python, Machine Learning, Predictive Modeling, Model Evaluation, Pandas, Data Visualization, Regression Analysis, Classification Models |  | | $0.0207 |
| fmegahed/shoulder_fatigue_modeling | gpt-6-sol |  | Python, Pandas, Machine Learning, Predictive Modeling, Regression Analysis, Classification Models, Model Evaluation, Data Visualization |  | | $0.0404 |
| fmegahed/tavr_paper | gemini-3.8-flash |  | Python, Machine Learning, Classification Models, Predictive Modeling, Model Evaluation, Pandas, Scikit-learn, Web Development |  | | $0.0177 |
| fmegahed/tavr_paper | gpt-6-sol |  | Predictive Modeling, Classification Models, Regression Analysis, Model Evaluation, Python, Pandas, Scikit-learn, Web Development |  | | $0.0324 |
| fmegahed/vaccines_spatial_and_optimization | gemini-3.8-flash | | | | AI_NoObjectGeneratedError: No object generated: could not parse the response. | |
| fmegahed/vaccines_spatial_and_optimization | gpt-6-sol | R, Regression Analysis, Statistical Inference | Predictive Modeling, Model Evaluation, Data Wrangling, Data Visualization, Statistical Analysis |  | | $0.0992 |
| fmegahed/covid19-deaths | gemini-3.8-flash | R, Clustering & Segmentation, Regression Analysis | Data Visualization, Data Wrangling, Model Evaluation, Statistical Analysis |  | | $0.0244 |
| fmegahed/covid19-deaths | gpt-6-sol | R, Clustering & Segmentation, Regression Analysis | Data Wrangling, Data Visualization, Statistical Analysis, Classification Models, Model Evaluation |  | | $0.0463 |
