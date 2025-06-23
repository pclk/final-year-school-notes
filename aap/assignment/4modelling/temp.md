
"""
Salary Prediction Application - Enhanced UX Design with AI Integration

## User Journey: Sarah's Experience with the Salary Prediction Application

To illustrate the enhanced UX, let's follow Sarah, a software developer with 5 years of experience, as she uses the salary prediction application.

**1. Initial Access & Onboarding:**

-   Sarah lands on the application's homepage, which features a clean, modern design and a clear call-to-action: "Predict Your Salary."
-   She notices the application is well-structured and visually appealing, instilling trust and professionalism.
-   If it's her first time, a brief, interactive onboarding tour highlights the key features and benefits of the application, emphasizing the AI-powered assistance and personalized insights.  This tour is optional and can be skipped.

**2. Stage 1: Interactive Form Experience**

-   **Adaptive Form Initiation:** Sarah clicks "Predict My Salary" and is greeted with the first section of the form: "Basic Information." The form uses progressive disclosure, showing only one section at a time, making it less overwhelming.
-   **Smart Input & Dynamic Validation:** In the "Job Title" field, as Sarah starts typing "Software De...", the AI suggests "Software Developer," "Software Engineer," and "Senior Software Developer" from a standardized list. She selects "Software Developer."  The "Years of Experience" field is a number input, and as she enters "5", a green checkmark appears, indicating valid input.
-   **Proactive AI Assistance:**  As Sarah moves to the "Skills" field in the "Professional Details" section, the AI proactively suggests skills relevant to a Software Developer, such as "JavaScript," "Python," "Cloud Computing," based on common skills for her job title and experience level.
-   **Contextual Tooltips & Explainers:**  Hovering over the "Location Preference" field, Sarah sees a tooltip powered by AI: "Considering a move? Explore cost-of-living comparisons for different locations in your results."
-   **AI Chat Interaction:**  In the "Location Preferences" section, Sarah is unsure about which locations to consider. She uses the AI chat assistant, which is context-aware.  Suggested questions include "Compare salaries in Austin vs. Seattle?" She clicks this suggestion and instantly gets a summarized comparison in the chat window, influencing her location choices in the form.
-   **Gamification & Progress:** A progress bar at the top of the form visually shows her advancement through the sections, motivating her to complete the process. Subtle animations provide positive feedback as she completes each section.

**3. Stage 2: Interactive Results Dashboard**

-   **Personalized Salary Cards:** After submitting the form, Sarah is taken to her personalized results dashboard.  Prominently displayed are "Salary Cards" for her preferred locations (US, UK, Canada). Each card shows a dynamic salary range: "Good," "Better," and "Best" case scenarios, along with a "Cost of Living Index" for each location.  Personalized insights below the cards state, "In Austin, TX, your Software Developer skills are in high demand, leading to potentially higher earning in the 'Better' to 'Best' salary range."
-   **Interactive Market Insights Panel:**  Sarah explores the "Market Insights" panel. The "Market Trends" chart is interactive; she zooms in to see salary trends for Software Developers over the last year and uses a time-range selector to focus on the last quarter.  In the "Skills Gap Analysis" radar chart, she sees a visualization of in-demand skills for her profession and how her skills align.
-   **Actionable Career Development Section:**  In the "Career Development" section, the "Skill Gap Analysis & Recommendations" provides specific advice: "Consider enhancing your skills in 'Cloud Security' and 'DevOps'.  Here are recommended courses on Coursera and Udemy to bridge this gap."  The "Career Path Visualizer" shows potential career progression paths for a Software Developer, including roles like "Team Lead," "Engineering Manager," and "Solutions Architect," with predicted salary growth for each path.
-   **Enhanced AI Career Advisor:**  Sarah notices the AI Career Advisor in the side panel. It proactively suggests, "Based on your profile, roles in 'Cloud Computing' are showing .significant salary growth. Explore companies in this sector in your preferred locations."  She uses the AI chat and asks, "What if I learn AWS and get certified?" The AI Advisor responds with scenario planning: "Gaining AWS certification and skills could potentially move your salary prediction towards the 'Better' or 'Best' range, especially in locations with a strong tech presence like Seattle or Bay Area."
-   **User Feedback & Iteration:**  Throughout the dashboard, Sarah sees quick feedback icons (thumbs up/down) for each section and insight.  She uses these to provide immediate feedback on the relevance and helpfulness of the information.  At the bottom of the dashboard, there's a clear "Feedback" button for more detailed comments and suggestions.

**4. Ongoing Engagement & Iteration Loop:**

-   Sarah finds the application incredibly helpful and saves her profile.  She receives an email summarizing her salary predictions and career insights, with a link to revisit her dashboard.
-   A week later, she receives a personalized email update: "Salary trends for Software Developers in Austin are up by 3% this month.  Check your updated salary prediction dashboard." This proactive communication encourages her to re-engage with the application and see updated insights.
-   The application team regularly reviews user feedback and usage data. They identify that many users are asking for more insights into remote work salary trends.  In the next design iteration, they add a "Remote Work Salary Insights" section to the dashboard, directly addressing user needs and improving the application's value.

**Summary of UX Improvements in Sarah's Journey:**

Sarah's user journey highlights how the enhanced UX principles translate into a practical and beneficial experience:

-   **Adaptive and Intelligent:** The form adapts to her inputs, and AI proactively assists her.
-   **Accessible and Inclusive:**  Sarah, as a screen reader user, can easily navigate and use the application.
-   **Personalized and Insightful:**  Salary predictions and career advice are tailored to her profile and preferences.
-   **Engaging and Motivating:**  Interactive elements and gamification keep her engaged and encourage completion.
-   **Efficient and Streamlined:**  The progressive form and clear dashboard provide information quickly and efficiently.
-   **User-Feedback Driven:**  Sarah's feedback and that of other users directly contribute to the application's ongoing improvement.

By focusing on these UX enhancements, the salary prediction application becomes not just a tool, but a valuable partner in Sarah's career planning journey.


## Components
The page has vertical carousel like animations, reminiscent of railway aesthetics. There are 4 stages, with the first being a walkthrough. we can use local storage to set true and false regarding whether they finished or skipped the walkthrough. Due to the extent of the experience, the experience progress, form fields and predicted results will be saved in localStorage. There is also a progress bar at the top detailing the stages. clicking on the stages will bring you back and forth.

the second and third will be the form experience. 

for the second there will be two fields: Job Title and Job Description. There will be a autocomplete of the next word for all items, as well as a "Refine description" for the job description.

the third will be autogenerated based on the job title and job description. It will feature fields like location, min_years_experience, soft skills hard skills etc. User can dial these details.

the final will be the results. there will be two parts: left and right. Left will include the final prediction across the 3 countries, Right will include an AI chatbot that allows you to communicate in regards to the results.

You will be able to restart the profile, or reset the conversation. localStorage will be cleared and user can restart at the second page.

You will be able to backtrack.
