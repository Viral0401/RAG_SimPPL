import dspy  # type: ignore

trainset = [

    # === MIT PKG Innovation Fellowship ===
    dspy.Example(
        question="What challenge is your MIT PKG project solving?",
        reasoning="The problem is that millions of women and girls in Bangladesh lack access to accurate, culturally relevant menstrual hygiene information. Existing physical interventions are effective but hard to scale. Their chatbot bridges this gap using WhatsApp and LLMs.",
        answer="We present a Bengali WhatsApp chatbot to deliver a digital literacy intervention for improving menstrual hygiene management awareness among local populations in Bangladesh. We use large language models grounded in verified and accurate information to support community-based participatory research in partnership with WaterAid Bangladesh."
    ).with_inputs("question"),

    dspy.Example(
        question="What makes your solution to menstrual health education effective?",
        reasoning="The chatbot is accessible through WhatsApp, supports Bengali, and is powered by RAG from verified health sources. It’s designed with safety, cultural sensitivity, and scale in mind.",
        answer="Our chatbot Sakhi is built on retrieval augmented generation and delivers responses grounded in trusted content from WHO, UNICEF, and WaterAid Bangladesh. Supporting Bengali and ‘Beng-lish,’ it ensures inclusivity. Deployed on WhatsApp, it provides safe, private, and interactive education with built-in guardrails."
    ).with_inputs("question"),

    dspy.Example(
        question="Who are your implementation partners in the MIT PKG project?",
        reasoning="WaterAid Bangladesh is the local deployment partner providing trust and reach in the community. They co-designed rollout strategies and signed an MoU. The team also consulted local developers with healthcare experience.",
        answer="We partner with WaterAid Bangladesh, whose staff have built strong relationships with the community. They assist with deployment, feedback collection, and provide domain-specific insights. We also engaged female Bengali-speaking app developers to ensure the chatbot design was community-informed."
    ).with_inputs("question"),

    # === Mozilla Senior Fellowship Application ===
    dspy.Example(
        question="What kinds of projects are Mozilla Senior Fellows expected to undertake?",
        reasoning="Projects should influence the trustworthy AI ecosystem and address social justice. Examples include AI regulation research, dataset transparency, and tech accountability initiatives.",
        answer="Senior Fellows are expected to pursue independent projects that build a more human-centered internet. These include research on AI regulation, dataset fairness, and open-source governance, intersecting with climate, labor, surveillance, or racial justice to shift systemic power structures."
    ).with_inputs("question"),

    dspy.Example(
        question="How should AI-related laws and regulations be assessed?",
        reasoning="Laws should be monitored using oversight mechanisms, enforcement strategies, and community-based input. The focus is on ensuring people’s rights are upheld through transparency and audits.",
        answer="AI-related laws and regulations should be assessed using robust oversight mechanisms, enforcement frameworks, community input, and audits. This ensures public accountability and safeguards users’ rights in real-world deployments."
    ).with_inputs("question"),

    dspy.Example(
        question="What is expected of a Mozilla Fellow during their term?",
        reasoning="Fellows must carry out an impactful independent project, engage with a cohort, and contribute to Mozilla’s ecosystem through public learning and community-based work.",
        answer="Fellows are expected to use the fellowship as a platform for independent work that furthers trustworthy AI, collaborate across disciplines and geographies, and publish their findings to foster open, inclusive digital environments."
    ).with_inputs("question"),

    # === exploreCSR Grant Proposal ===
    dspy.Example(
        question="What impact has the exploreCSR grant had on your past programs?",
        reasoning="The grant enabled a fellowship program, allowing students to publish, deploy social-good solutions, and even incubate startups like Sakhi. It bridged research training and real-world deployment.",
        answer="The exploreCSR grant allowed us to scale responsible computing fellowships across underserved colleges in India. Students created research-backed tools deployed with UN and nonprofits. One group built Sakhi, a menstrual health chatbot now incubated at MIT. Fellows went on to mentor NYU and MIT students in capstone projects."
    ).with_inputs("question"),

    dspy.Example(
        question="How does your program center marginalized students?",
        reasoning="The program focuses on underserved communities, offering stipends, practical training, and mentorship. It builds computing capacity through real-world impact.",
        answer="We center historically marginalized groups by recruiting students from Tier II and III colleges, offering stipends, providing access to mentorship and compute resources, and helping them lead socially impactful research. Our fellows now mentor globally and present at top research venues."
    ).with_inputs("question"),

    dspy.Example(
        question="What are the core goals of your exploreCSR initiative?",
        reasoning="The goals are to develop inclusive research capacity, scale student-led social computing projects, and integrate responsible computing into departmental culture.",
        answer="Our core goals are to build student capacity in computing research, promote responsible and inclusive innovation, and integrate community-based, real-world challenges into the academic curriculum to drive systemic change."
    ).with_inputs("question"),

]
