import dspy # type: ignore
from dspy.teleprompt import BootstrapFewShot # type: ignore
from grant_cot_module import GrantCoTModule
from grant_cot_examples import trainset
import streamlit as st

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
# ✅ ADD THIS: Configure the LLM
lm = dspy.LM("openai/gpt-4o", api_key=OPENAI_API_KEY)
dspy.configure(lm=lm)

# ✅ Metric for optimization (simple match)
def metric(example, prediction, trace=None):
    return int(example.answer.strip().lower() in prediction.answer.strip().lower())

# Initialize your module
cot = GrantCoTModule()

# Compile with BootstrapFewShot
teleprompter = BootstrapFewShot(metric=metric, max_bootstrapped_demos=2, max_labeled_demos=2)
optimized_cot = teleprompter.compile(cot, trainset=trainset)

# ✅ Save as .json (for streamlit use)
optimized_cot.save("optimized_grant_cot.json")
