import dspy # type: ignore

class GrantCoTModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought("question -> reasoning, answer")

    def forward(self, question):
        return self.cot(question=question)
