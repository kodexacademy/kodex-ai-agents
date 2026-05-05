#  LLMs in Loop - Document Correction Loop | Kodex Academy
Code example in AI agents series. Here I use Groq LLM.

**Links:**
- [Video link]()
- [Blog link]()

## How to run this example

1. Clone this repo
2. Navigate to downloaded folder and create new virtual environment
```
python -m venv llm-loop
```
3. Activate virtual environment
```
# mac/linux
source llm-loop/bin/activate

# windows
llm-loop\Scripts\activate
```

4. Environment variables
Create .env file for environment variables. Add following variables
- GROQ_API_KEY=<GROQ_API_KEY>
- GROQ_LLM_MODEL=llama-3.3-70b-versatile

5. Install dependencies
```
pip install -r requirements.txt
```

6. Run script
```
python llm_loop_agent.py
```
