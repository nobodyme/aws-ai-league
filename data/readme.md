# qna crafter

US State government, Local and Education - understand citizen needs in plain language. Handle complex scenarios like "I want to open a food truck" or "My neighbor's tree fell on my property" while providing step-by-step guidance through permit applications and licensing. Datasets generated should include realistic user queries and appropriate concise and helpful answer like an assistant from US gov public services

Suggestions add in:
Refusals: High-quality denials with helpful redirection (no legal verdicts, no identity lookups, no neighbor doxxing).

Add jurisdiction metadata: state, county/city, agency, program_name, form_names (even if placeholder), last_verified_date, phone/url placeholders. Teach the model to say “varies by city—verify here”.

Time sensitivity flags: is_time_sensitive, renewal_cycle, deadline_months, fees_change_often. Push the model to warn when rules change.

Spanish


All of the above failed, should revist refusals

- Try another model for data generation
- Refuse irrevelant topics
- Ability to ans who are you type of questions