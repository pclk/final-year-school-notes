# AI Ethics Analysis Assignment

Students are required to research one AI company (Adobe, OpenAI, Google, Microsoft or etc) that used generative AI and analyze its practices and policies through the lens of AI ethics. The analysis should cover various ethical dimensions, including data privacy, bias and fairness, transparency, accountability, and the societal impact of AI technologies. Write a report on the relevance of AI ethics for the organization that you have researched on.

## Requirements
- The analysis should be 1500-2000 words in length.
- Use at least 5 credible sources, including academic papers, industry reports, and official company documents.

## Report Structure and Marking Criteria

### Company Analysis (3 marks)
**Marks: 2.4 – 3.0**
- Provides a detailed and insightful analysis of the chosen AI company's practices and policies.
- Covers multiple ethical dimensions with in-depth evaluation.

OpenAI was the first to kickstart the Generative AI (GenAI) race. From a non-profit research laboratory in 2015, to an influential force that has reshaped how we think about AI's capabilities. However, because of their first mover advantage, there is a lot of eyes looking over them.

Before OpenAI, there was already a pursuit of human-like intelligence in machines. Many attempts such as the Eliza chatbot by Alan Turing have been made in the past. Therefore, the concept of an intelligent machine going rogue have been ingrained into our minds, with countless movies made such as the terminator. Steven Hawking even told us that the biggest threat to humanity is Artificial Intelligence.

The company's release of ChatGPT in the late 2022 catalyzed unprecedented public engagement regarding GenAI. OpenAI started as a non-profit entity, but to pursue technological advancements, have to undergo leadership changes, move to a capped-profit and eventually a for-profit (https://www.theguardian.com/technology/2024/sep/26/why-is-openai-planning-to-become-a-for-profit-business-and-does-it-matter#:~:text=What%20are%20the%20changes%20OpenAI,in%20the%20profit%2Dmaking%20business.).

OpenAI has participated within global AI policy discussions. For example, the proposal for America to increase investments in energy and infrastructure to bolster the American economy, including OpenAI's official documents (https://openai.com/global-affairs/openais-economic-blueprint/), the Stargate project (https://openai.com/index/announcing-the-stargate-project/), and lobbying against EU AI regulations (https://time.com/6288245/openai-eu-lobbying-ai-act/)


### Data Privacy and Security (3 marks)
**Marks: 2.4 – 3.0**
- Thoroughly examines the company's approach to data privacy and security.
- Provides specific examples and evaluates effectiveness.

OpenAI claims that they give their users the choice to control whether their content can be used to train AI. They provide instructions on how to opt-out of these collection of personal data (https://help.openai.com/en/articles/5722486-how-your-data-is-used-to-improve-model-performance) for their ChatGPT, Sora and Operator applications.

Their privacy policy (https://openai.com/policies/privacy-policy/) is short and easy to read. To summarize, they collect IP address, log data, usage data, device information, location, and other cookie information. They may use it to improve their services, commuicate with you, prevent fraud and comply with legal obligations. They may disclose it to vendors and service providers, business transfers, government authorities and affiliates. Regarding data retention, its not as clear, but some data are adjustable by the user, like chat history. Other data depends on factors that the user cannot control, like OpenAI's purpose for the data, risk of harm, and legal requirements.

For API usage (https://platform.openai.com/docs/models/how-we-use-your-data#how-we-use-your-data), most of the data retention last around 30 days, with some being deletable by customer and some not being retained.

Their data security can be reviewed here (https://trust.openai.com/). They have certifications for SOC 2, SOC 3, GDPR etc.

No matter how their privacy policy is structured, it cannot be compared to the privacy of locally hosting an open-sourced model like llama-3.3 or deepseek-v3. Therefore, their data privacy should be considered good but not the best.

### Bias and Fairness (3 marks)
**Marks: 2.4 – 3.0**
- Comprehensive analysis of how the company addresses bias and fairness in its AI systems.
- Includes detailed examples and critical evaluation.

The tricky situation about evaluating bias within a company's models is that if bias is found, the media will be loud about it. But if there's no bias, the media wouldn't notice it. That being said, there were some instances which the public discussed about the existing bias in OpenAI's models.

OpenAI's models bias includes DALL-E's CLIP model performing poorly on low-income and non-Western life styles (https://arxiv.org/abs/2311.05746), GPT-3's bias on various human trait (https://arxiv.org/pdf/2005.14165) (Page 36: Gender, Race, Religion). 
In the study, Assessing the potential of GPT-4 to perpetuate racial and gender biases in health care: a model evaluation study (https://arc.net/l/quote/oekugact), it is found that GPT-4's modelling of disease prevalence by race had signficiant differences compared to the true US prevalence estimates.
OpenAI claims to be "Safeguarding against bias"(https://openai.com/index/openai-safety-update/), with a combination of content moderation, safety filters, preventing image stereotyping, vocal stereotyping and conducting ongoing research on bias and fairness with AI.

Though they claim to be fighting against bias, it seems that the research point towards the fact that their models do indeed contain bias. It is understandable that most generative AI contain such biases, and they may be doing their best to prevent it. Thus, their bias and fairness should be considered moderate.

### Transparency and Accountability (3 marks)
**Marks: 2.4 – 3.0**
- In-depth discussion on the company's transparency and accountability measures.
- Provides specific examples and critical evaluation.

OpenAI's Transparency and Accountability they provide system cards for all of their models (https://openai.com/index/openai-o1-system-card/). This includes details like their Model data and training, observed safety challenges, Jailbreak evaluations and so on.

OpenAI however, has documents regarding their preparedness framework, (https://cdn.openai.com/openai-preparedness-framework-beta.pdf), which helps with studying catastrophic risks from AI. They also shared 10 practices that they actively use and improve upon (https://openai.com/index/openai-safety-update/). Additionally, there are details regarding their safety practices within their safety portion of the website (https://openai.com/safety/)

In terms of accountability, they claim to have external red teaming (https://arc.net/l/quote/iphfygum)(o1), (https://arc.net/l/quote/svocctdq)(4o) for their models. They also have a usage policy (https://openai.com/policies/usage-policies/) that provides guidelines and recommendations with those building with ChatGPT.

Generally their transparency and accountability is pretty strong. They have good documentation on the decision processes behind the design of their models using system cards. They also share their research (https://openai.com/research/). Thus, their transparency and accountancy should be considered excellent.

### Societal Impact (1 mark)
**Marks: 0.9 – 1.0**
- Thorough analysis of the societal impact of the company's AI technologies.
- Includes specific examples and critical evaluation.

OpenAI shares their societal impact in their "News" section of their website. Some examples include Strengthening America’s AI leadership with the U.S. National Laboratories (https://openai.com/index/strengthening-americas-ai-leadership-with-the-us-national-laboratories/), Morgan Stanley uses AI evals to shape the future of financial services (https://openai.com/index/morgan-stanley/), and Bertelsmann powers creativity and productivity with OpenAI (https://openai.com/index/bertelsmann-powers-creativity-and-productivity-with-openai/).

On a larger scale, we can see that OpenAI's societal impact is in enhancing productivity and efficiency in automating tasks like customer service, data analysis etc. The nature of GenAI also allows for democratization of access to information, making it easier for students and individuals to retrieve information, and use Web-powered generative AI tools, generate custom images, videos and so on, which was once restricted by skill, resource, and time barriers.

Additionally, education receives a boost. Typically students who study math require guidance when they don't understand a math problem. Such is an issue even during the internet era, as teachers and lecturers have their own time to spend and may not attend to user questions immediately. However with GenAI, students can receive a personal math tutor that can receives guidance. For example, with Khan academy (https://openai.com/index/khan-academy/).

Though artists may be unhappy with Image generation and how it affects their prospects, they ultimately are not harmed by it, and instead may need to undergo a change in their processes. Like painters who were unhappy with the invention of film cameras, this is the byproduct of increased efficiency of our tools.

Overall, the societal impact is rather positive with little harm. Thus, their societal impact should be considered excellent.

### Conclusion 

As a leader in the field and the company who started the Generative AI race, OpenAI has led in many developments in AI. From GPTs, to popularizing function calling, its clear that OpenAI has a responsibility to not just lead in model capabilities, but also shape the ethical framework that governs impact its impact on society.

Overall, with data privacy and security being good, bias and fairness being moderate, transparency and accountability being excellent, and societal impact being excellent, it can be concluded that OpenAI has an overall rating of good when it comes to AI Ethics.



**1. Ethical AI Development**

*   **Core Principle:** OpenAI's mission centers on developing AI that benefits humanity. They strive to create AI technologies that are safe, secure, and aligned with human values. [2]
*   **Safety and Control:**  This involves rigorous testing, safety research, and building AI systems that are robustly controlled. [2]
*   **Initiatives for Responsible AI:** OpenAI actively engages in research and development to address bias, promote transparency, and understand the societal impacts of AI. They aim to ensure AI systems align with human values and benefit everyone. [1]
*   **Preparedness Framework:** OpenAI has adopted a preparedness framework to measure and forecast potential risks associated with AI development. They commit to halting deployment if safety mitigations lag behind. [9]
*   **Independent Oversight:** The Safety and Security Committee (SSC) has been elevated to an independent board oversight body, with the authority to delay model releases if safety concerns are not adequately addressed. [15, 13]

**2. Transparency and Openness**

*   **Commitment:** OpenAI aims to be transparent in its research and development processes. [2]
*   **Sharing Information:** They publish research papers, share insights, and engage with the broader AI community to foster collaboration and knowledge exchange. [2]
*   **Balancing Act:** OpenAI balances openness with security, ensuring sensitive technologies don't pose risks to society. [2]
*   **Transparency Note:**  Technical recommendations and resources are provided to help customers use Azure OpenAI models responsibly, following the Microsoft Responsible AI Standard. [12]
*   **Limitations:**  Despite this commitment, some critics argue that OpenAI has not always been fully transparent about the data used to train models, leading to concerns about copyright infringement. [23] Additionally, some of their models are considered "black box" systems, making it hard to understand the root causes of their behavior. [6, 29]

**3. Bias and Fairness**

*   **Recognition of the Issue:** OpenAI acknowledges the potential for bias in algorithms. [1, 6]
*   **Mitigation Efforts:** They invest in research to develop fair and unbiased algorithms, addressing disparities in data that could lead to discriminatory outcomes. [1]
*   **Fairness-Aware Algorithms:** OpenAI pioneers research in algorithms designed to minimize biases across different demographic groups. [17]
*  **Diverse Data:** Prioritizing diversity in training datasets is crucial, along with regular audits to detect and rectify biases. [11]
*  **Red Teaming:** Models undergo "red teaming" to probe vulnerabilities and identify potential biases. [6]
*   **Tools:** OpenAI provides tools like Fairlearn and Fairness Scorecard to help developers recognize and mitigate bias in AI applications. [11]
*  **Ongoing Challenge**: It's important to note that no AI system is entirely free from bias, and addressing bias is an ongoing challenge for the entire industry. [16]
*   **Name-Based Bias Research**: OpenAI has conducted research analyzing how ChatGPT responds based on user names to understand underlying biases. [22]

**4. Data Privacy and Security**

*   **Encryption:** OpenAI uses robust encryption standards (AES-256 for data at rest, TLS 1.2+ for data in transit) to protect data. [3, 21]
*   **Access Controls:** Strict access controls limit who can access sensitive information. [3]
*   **Third-Party Audits:** OpenAI's API, ChatGPT Enterprise, ChatGPT Team, and ChatGPT Edu products are evaluated under a SOC 2 Type 2 report to align with industry standards. [3]
*   **Data Handling:** OpenAI is committed to complying with data privacy regulations like GDPR and CCPA. [3]
*   **User Data for Training:** OpenAI uses client data for training its models, which has raised concerns. [3] However, they also provide customizable data retention settings. [3]
*  **Data Retention:** Workspace admins control how long data is retained. Deleted conversations are removed within 30 days, unless legally required to be kept. [21]
*   **Data Storage Location:** OpenAI's data storage locations are not fully disclosed. [3]
*  **Additional Data Leaks:** Custom GPTs interacting with external APIs can potentially transmit conversation text to those APIs, which are outside of OpenAI's control. [3]
*   **User Rights:** Users can exercise privacy rights by submitting requests to privacy.openai.com or dsar@openai.com. [25]
*  **Data Center Security:** OpenAI is implementing stringent controls and advanced security for its data centers, to address both insider and outsider threats. [10]
* **Network and Tenant Isolation**: OpenAI is building isolated network environments for different AI systems and tenants to prevent unauthorized access and data breaches. [10]

**5. Accountability**

*   **Mechanisms for Addressing Issues:** OpenAI has mechanisms in place to identify and rectify potential issues or biases, including thorough auditing (internal and external). [1]
*  **Feedback:** They are responsive to feedback from users, researchers, and the public. [1]
*   **Adaptation and Dialogue:** OpenAI engages in ongoing dialogue with the public, policymakers, and experts to incorporate diverse viewpoints. [1]
*   **Continuous Improvement:** They are dedicated to refining their ethical guidelines and evolving their practices to align with the evolving landscape of AI ethics. [1]
*   **Accountability Letter:** A group of signatories has published an accountability letter demanding that OpenAI be held responsible for its past and future actions, and suggesting mechanisms for increased accountability. [19]

**6. Collaboration and Partnership**

*   **Emphasis on Collaboration:** OpenAI emphasizes collaboration with other organizations, governments, and stakeholders to address ethical, safety, and governance challenges in AI. [2]
*   **Global Alignment:** These collaborations aim to ensure that AI development aligns with global interests and contributes to the common good. [2]

**7. Potential Risks and Concerns**

*   **Misuse of AI:** OpenAI platforms can produce inappropriate content, deceptive deepfakes, and even aid in launching cyberattacks. [7]
*   **Environmental Impact:** AI development has environmental impacts due to electronic waste and the significant amount of water required for cooling. [7]
*   **Data Usage Concerns:** There are concerns about improper data usage, poor labor conditions, and other risks posed by OpenAI. [7]
*   **Lack of Transparency:** Some accuse OpenAI of a lack of transparency regarding what data is used to train their models and what safety measures are in place. [23, 14]
*  **Bias in Generative Outputs**: Generative AI systems can perpetuate stereotypes through text and images. [6]
*   **Ethical Dilemmas:**  There are ethical concerns about AI replacing human interaction and the potential for AI to degrade critical thinking skills. [5]
*   **Data Privacy Issues:** Data privacy concerns remain, including the potential exposure of data sent through the API and compliance with GDPR. [14]
*   **Financial Incentives:**  Some concerns have been raised regarding the shift from a non-profit structure to a capped-profit model, and the impact on the mission of safe AI development. [7, 19]
*   **Regulatory Scrutiny:** OpenAI has faced scrutiny and fines from data protection authorities, such as the Italian Garante, for data privacy violations. [18]

**8. Specific Measures Implemented**

*   **Safety and Security Committee:** Established to oversee critical safety and security measures and conduct reviews of safety processes. [13]
* **Trusted Computing**: Implementing trusted computing for AI hardware to create a secure environment for AI. [10]
* **Auditing and Compliance Programs**: Implementing AI-specific audit and compliance programs to ensure integrity and security of AI systems. [10]
* **Bug Bounty Program:** A bug bounty program is in place to help identify security vulnerabilities. [14, 21]
*   **Usage Policies:** Updated usage policies to be more readable, adding service-specific guidance. [28]

**In conclusion:**

OpenAI is making significant efforts to address the ethical challenges of AI development, including promoting transparency, mitigating bias, and ensuring data privacy. However, as with any rapidly evolving technology, there are ongoing risks and concerns. It's critical for OpenAI to continuously adapt, engage with stakeholders, and prioritize ethical principles in its pursuit of AI innovation. The independent oversight provided by the SSC, as well as the increased scrutiny by regulators, will likely play a crucial role in guiding OpenAI towards a responsible and beneficial future for AI.

2. [analyticsvidhya.com](https://www.analyticsvidhya.com/blog/2023/12/openai-prepares-for-ethical-and-responsible-ai/)
3. [rtinsights.com](https://www.analyticsvidhya.com/blog/2023/12/openai-prepares-for-ethical-and-responsible-ai/)
4. [medium.com](https://medium.com/@gargg/guiding-the-future-of-ai-an-inside-look-at-openais-ethical-policies-3503f17b6b0b)
5. [bullandbearmcgill.com](https://bullandbearmcgill.com/openai-the-future-and-ethics-of-artificial-intelligence/)
6. [chatbase.co](https://www.chatbase.co/blog/is-openai-safe)
7. [medium.com](https://ryanranas.medium.com/is-openai-inherently-biased-05d42f2449c5)
8. [maginative.com: OpenAI Updates Safety and Security Measures with Independent Oversight](https://www.maginative.com/article/openai-updates-safety-and-security-measures-with-independent-oversight/)
10. [analyticsvidhya.com: Forget Firewalls: 6 OpenAI Security Measures for Advanced AI Infrastructure](https://www.analyticsvidhya.com/blog/2024/05/openai-security-measures/)
11. [justthink.ai: OpenAI Policy Manager: How to Mitigate Bias in AI](https://www.justthink.ai/blog/openai-policy-manager-how-to-mitigate-bias-in-ai)
12. [athensjournals.gr: The Ethical Dilemma with Open AI ChatGPT: Is it Right or Wrong to prohibit it?](https://www.athensjournals.gr/law/2024-10-1-6-Coltri.pdf)
14. [medium.com: Ensuring Privacy and Data Safety with OpenAI](https://medium.com/@mikehpg/ensuring-privacy-and-data-safety-with-openai-a-comprehensive-guide-5a744e2c6416)
15. [openai.com: An update on our safety & security practices](https://openai.com/index/update-on-safety-and-security-practices/)
16. [viso.ai: Ethics in AI – What Happened With Sam Altman and OpenAI](https://viso.ai/deep-learning/ethics-in-ai-sam-altman-and-openai/)
17. [signitysolutions.com: Addressing Bias in AI: Strategies for Bias Mitigation with OpenAI](https://www.signitysolutions.com/tech-insights/addressing-bias-in-ai-mitigation-strategies-with-openai)
19. [openai.com: Enterprise privacy at OpenAI](https://openai.com/enterprise-privacy/)
21. [openai.com: Privacy policy](https://openai.com/policies/row-privacy-policy/)
22. [openailetter.org](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAYygrcRZ9oVEH7A9mZU8H0doyJgfKd6VTwToFF8vkhu3hJUGKem1qIRv04pcJBBzt_OXLROqKH2n2rcIp9CteI_s8mlGTibQX5DjoUbbPmW2_Y30gvyQiReLth0%3D)
23. [lewissilkin.com](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAYygrcStdtmrMtlyzR4maLWqGFvMet2x816j5hQYi8g5hQeJoERpX1PN_leYmwx8qy5wqDoOlEwUBY5qXP-5mdcFNS67AZXaZHdrIKqQh599hfwRnfOwhI4ghEEdUR6C2kr-B38JIGLLX3kLF8EFa2p34uPDZhYoTuxiXbKDQKxIR4DFPiv3GNe-pmQcxpSMCXliYcOe_ZI3bIzekY89oAz0G6EPvFiXWakDm9xfawXpDFiSP0EYag%3D%3D)
24. [neubrain.com](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAYygrcQdZEat-vsd5kgJWX9VLq13Kh-S89t5h0mZR2xejKgq7o-6sBXuqonNJrPbplPlcCDraoy7gK0-YtHSwazNLZy-zgr7F80oKgnBLOODBfaYyXcXrbUyuxfZ1-mQD18iADx80g9vyfc5M4fk_EiNAdomZFw6Xvos1N31YLOx5XXutv6xnxEPWtqjfdyLEKODTqHFf7i-ebz-k0CEFgn2w-rRNJEMXrfx3yg%3D)
25. [softkraft.co](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAYygrcQL6ZL5tMTuewkANuyRKV56ITlPcMth-1RE0Xk5JfoMimhxTAT3JJ6nM9y_NZzloBuQq9rituKCpKXin40TupX7eQOFJyqJfdhjvM0Rv1rR_ATGW3Gdu_A9Z6CSGIYvP0LYnJ_ujcDA2A%3D%3D)
26. [openai.com](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAYygrcS6OSQNRSghOp1gzlqGtajrIpfTqQ8YlubMhaaiffe9PcPP8y3932IHil_s-LXwNV2cYbWmTW42g-n7T6mELTnHewnRUd2G-UK4DYyZAbRWU7xr7XDrIxDyLLwkB5JrCOeupQYiFQ%3D%3D)
