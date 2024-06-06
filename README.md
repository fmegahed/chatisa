# ChatISA

> ChatISA is your personal, free, and prompt-engineered chatbot, where you can chat with one of nine LLMs. The chatbot consists of four main pages: (a) Coding Companion, (b) Project Coach, (c) Exam Ally, and (d) Interview Mentor.

<img width="1433" alt="image" src="https://github.com/fmegahed/chatisa/assets/22730186/3a0c2839-3384-428c-9aa5-e0cf95ba1296">

## What made this project possible?

### Contributors

* [Fadel Megahed](megahefm@miamioh.edu)
* [Joshua Ferris](ferrisj2@miamioh.edu)

### Support & funding

* Farmer School of Business
* US Bank

### References and inspiration

* Prompt Engineering: Adapted from [Assigning AI by Mollick and Mollic 2023](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4475995).
* Streamlit App: Adapted from [ChatGPT Apps with Streamlit](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-chatgpt-like-app).
* Creating multi-page apps from a [Stack Overflow answer](https://stackoverflow.com/a/74418483) and the following [GitHub project](https://github.com/jiatastic/GPTInterviewer/blob/main/Homepage.py).

## Contributing

Contributions are warmly welcomed. To contribute, please fork this repo and create a pull request with your changes.

## Setting up your environment

These setup instructions assume you are using [conda](https://conda.io/projects/conda/en/latest/index.html#).

```sh
conda env create -f environment.yaml
conda activate chatisa
```

## Running the app

```sh
streamlit run chatgpt.py
```
