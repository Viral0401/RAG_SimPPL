import dspy  # type: ignore

trainset = [

    # === MIT PKG Innovation Fellowship ===
    dspy.Example(
        question="What challenge is your MIT PKG project solving?",
        reasoning="Let's think step by step. The applicant describes that menstrual hygiene misinformation is a serious issue in Bangladesh. They aim to deliver verified health content via WhatsApp in Bengali.",
        answer="We aim to improve menstrual hygiene awareness in Bangladesh using a Bengali WhatsApp chatbot grounded in accurate information."
    ).with_inputs("question"),

    dspy.Example(
        question="What makes your solution to menstrual health education effective?",
        reasoning="The effectiveness comes from verified content delivery via WhatsApp, designed with WaterAid and community input.",
        answer="Our chatbot delivers verified menstrual health information through WhatsApp, combining LLMs and local partnership with WaterAid for impact."
    ).with_inputs("question"),

    dspy.Example(
        question="Who are your implementation partners in the MIT PKG project?",
        reasoning="The team has collaborated with WaterAid Bangladesh for field-level access and content credibility.",
        answer="We’re working with WaterAid Bangladesh as our local implementation partner to enhance outreach and trust."
    ).with_inputs("question"),

    # === Mozilla Senior Fellowship Application ===
    dspy.Example(
        question="What kinds of projects are Mozilla Senior Fellows expected to undertake?",
        reasoning="Mozilla seeks systemic interventions in AI that promote inclusion, transparency, and empowerment.",
        answer="Senior Fellows are expected to undertake independent projects that advance trustworthy AI and intersect with social justice, transparency, and inclusion."
    ).with_inputs("question"),

    dspy.Example(
        question="How should AI-related laws and regulations be assessed?",
        reasoning="Assessment involves oversight systems, enforcement mechanisms, and ensuring rights are upheld.",
        answer="AI laws should be assessed through robust oversight mechanisms, audits, public reporting, and community-based feedback."
    ).with_inputs("question"),

    dspy.Example(
        question="What is expected of a Mozilla Fellow during their term?",
        reasoning="Fellows are expected to commit to independent work, participate in cohort events, and share their work publicly.",
        answer="Fellows are expected to participate in cohort gatherings, collaborate on shared learning, and produce publicly available work."
    ).with_inputs("question"),

    # === exploreCSR Grant Proposal ===
    dspy.Example(
        question="What impact has the exploreCSR grant had on your past programs?",
        reasoning="The grant allowed scaling fellowships, expanding to underserved groups, and deploying global impact solutions.",
        answer="The exploreCSR grant enabled us to scale responsible computing fellowships, support underserved students, and deploy solutions with nonprofits and global partners."
    ).with_inputs("question"),

    dspy.Example(
        question="How does your program center marginalized students?",
        reasoning="The proposal highlights equitable practices such as stipends, local mentors, and nontraditional learning formats.",
        answer="We prioritize underserved students by offering stipends, mentorship, and accessible formats to support their participation."
    ).with_inputs("question"),

    dspy.Example(
        question="What are the core goals of your exploreCSR initiative?",
        reasoning="The core goals involve capacity-building, research training, and real-world deployment for societal good.",
        answer="Our goals are to build student capacity in computing research, foster inclusive mentorship, and deploy solutions in socially impactful domains."
    ).with_inputs("question"),

]
