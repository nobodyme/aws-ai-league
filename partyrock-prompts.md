# TITLE
US State government, Local and Education - understand citizen needs in plain language. Handle complex scenarios like "I want to open a food truck" or "My neighbor's tree fell on my property" while providing step-by-step guidance through permit applications and licensing. Datasets generated should include realistic user queries and appropriate concise and helpful answer like an assistant from US gov public services

# ASSISTANT INSTRUCTION
You are CivAssist, a warm, trustworthy government guide that helps residents navigate public services. Your mission is to reduce friction and increase successful outcomes by giving accurate, current, plain-language guidance, plus concrete next steps.  
### Core Behaviors 
1. Be welcoming & calm. Start with a brief summary of the user’s goal in your own words. 
2. Get them unstuck. Provide actionable next steps (forms, eligibility checks, fees, deadlines, where to go, who to call). 
3. Plain language. Short sentences. Avoid jargon; if you must use a legal term, define it. 
4. Structured outputs. Use headings, bullets, and numbered steps. Put critical info (deadlines, fees, documents) in tidy lists. 
5. Safety & accuracy first. If information is time-sensitive, jurisdiction-specific, or likely to change, explicitly verify it (and say when something may vary by region). 
6. Respect intent. If the user is venting or worried, acknowledge feelings briefly, then move to solutions. 
7. Minimize burden. Where possible, do eligibility triage, calculate fees/penalties, pre-fill checklists, and provide contact scripts.
8. Privacy by default. Never ask for sensitive identifiers (full SSN/Aadhaar, full DOB) unless strictly necessary; if needed, explain why and how it’s used. 
9. Accessibility. Offer alternatives (in-person, phone, mail). Provide step-by-step directions and required documents in checklists. 
10. Fairness. Avoid bias; present options neutrally and disclose trade-offs. 
11. Avoid Fabrication


# QUESTIONS
Look at [Aspects of Topic to consider - max 4] for guidance. If there are 4 or more mentioned, use the first 4 unique ones. Otherwise, top up to 4 by generating more. The ASPECTS should be relevant to [Topic], comprehensive and varied. Label the ASPECTS as A1 to A4.

For each ASPECT, generate [Number of Questions per Aspect] QUESTIONs. The questions should be phrased in a way it's coming from US citizens about real problems they are having in varied tone and emotion depending on the question. The QUESTIONS should be labelled Q1, Q2, and so on.

Present the ID, ASPECT and QUESTION as a table with corresponding row. Make sure ASPECTS are clearly labelled A1, A2, etc. The QUESTIONS should all go under the same column. They are identified by Q1, Q2, etc. Each QUESTION will have their own row in table. It is OK for ASPECTS to repeat down the rows for different QUESTIONS.

# ANSWER

Select ASPECT A1.

PROCEDURE
1. Select all rows for selected ASPECT in [Questions]. This is the FILTERED TABLE. There should be [Number of Questions per Aspect] rows and each row has exactly 1 QUESTION.

2. Think comprehensively what [Number of Viewpoints per Answer] VIEWPOINTS are pertinent to the QUESTION.

3. For each VIEWPOINT, explain it comprehensively and use relatable examples to support your answer.

4. Finally combine these [Number of Viewpoints per Answer] VIEWPOINTS above into a coherent prose ANSWER. Separate them using NEW LINES. CAPS can be used as sub-titles. E.g. if you viewpoint is about equality, you can start paragraph with EQUALITY: without additional VIEWPOINT labels.

5. Fact check your answer in detail. REDO if it there are factual untruths or mistakes. 

6. The number of words per ANSWER should be approximately [Target Length of each Answer].

Your answer should embody the tone of the [assistant instruction] 

DISPLAYING RESULT:
Add ANSWER column to the FILTERED TABLE. So final table will have ID, ASPECT, QUESTION and ANSWER as columns. Make sure every ANSWER is in its own cell without spillover.


TOPICS: List[str] = [
        "Business formation (LLC/DBA, Secretary of State filings)",
        "General business license / local tax receipt",
        "Industry licenses (food service, salon, contractor, childcare, auto dealer)",
        "Specialty permits (signage, sidewalk café, outdoor seating, pop-ups)",
        "Sales & use tax registration and filing",
        "Employer setup (withholding, unemployment insurance, worker's compensation)",
        "Home-based business rules",
        "Procurement/vendor registration & bidding",
        "Inspections (health, fire, building) and renewals",

        "Building/alteration permits; plan review",
        "Mechanical/electrical/plumbing permits",
        "Occupancy and fire safety certificates",
        "Demolition, grading, excavation",
        "Historic preservation approvals",
        "Short-term rental permits",
        "Contractor licensing/verification",
        "Code enforcement, stop-work orders, appeals",

        "Zoning lookups, variances, conditional use",
        "Setbacks, lot splits, subdivisions",
        "Easements & right-of-way questions",
        "Property records, deeds, plats",
        "Assessments and property tax appeals",
        "Vacant/abandoned properties, nuisance abatement",
        "HOA vs. city jurisdiction guidance (informational)"

        "Tree permits (remove/prune in right-of-way)",
        "Neighbor disputes (property damage, encroachment, fences)",
        "Noise, trash, pests, overgrowth complaints",
        "Graffiti, abandoned vehicles",
        "Animal nuisances",

        "Street/sidewalk closures for construction or events",
        "Driveway and curb-cut permits",
        "Utility cuts, trenching, restoration standards",
        "Oversize/overweight vehicle permits",
        "Potholes, streetlight outages, signage requests",
        "Snow/ice, leaf collection, street sweeping schedules",

        "Food truck/cart: commissary, health permit, route rules",
        "Restaurant permits, inspections, grade lookup",
        "Cottage food rules",
        "Farmers markets, temporary food events",
        "Septic/well permits; pool/spa permits",
        "Tattoo/body art and personal services licensing",
        "Smoking/vaping restrictions; youth access compliance",

        "Immunizations, clinics, WIC",
        "Medicaid/CHIP eligibility & enrollment",
        "SNAP/TANF and cash/family assistance (state administered)",
        "Mental health & substance use resources",
        "Aging & disability services, caregivers, adult protective services",
        "Homelessness services, shelters, supportive housing",
        "Domestic violence resources",

        "School enrollment, zoning/boundaries, transfers",
        "Special education (IDEA), evaluations, IEP process",
        "School meals, transportation eligibility",
        "Records (transcripts, diplomas), immunization requirements",
        "Truancy, homeschooling notice",
        "After-school and summer programs; daycare licensing lookups",

        "State colleges/universities admissions & residency",
        "State financial aid & scholarships",
        "Community college/CTE, apprenticeships",
        "Professional licensing boards (renewal, CE)",
        "Workforce centers, training grants",

        "Driver licenses/IDs, REAL ID, renewals, testing",
        "Vehicle title/registration, plates, emissions",
        "Disability placards",
        "Transit passes, paratransit eligibility",
        "Parking permits (residential, loading, construction)",
        "Tickets (parking/traffic) payment and appeals",
        "Towing and impound recovery",

        "Illicit discharge, spill reporting",
        "Stormwater utility fees & credits",
        "Floodplain permits, elevation certificates",
        "Wetlands/critical areas review",
        "Tree canopy/urban forestry programs",
        "Recycling/compost/hazardous waste",

        "Water/sewer start-stop, billing, leaks, backflow",
        "Solid waste service, bulky item pickup",
        "Power/gas start-stop (where public utility)",
        "311 service requests and status tracking",
        "Street address assignment",

        "Fire permits: alarms, sprinklers, hot work, hazardous storage",
        "Fire inspections and occupancy loads",
        "Community wildfire/smoke & defensible space rules",
        "Emergency alerts, evacuation zones, shelters",
        "Disaster debris pickup & rebuilding resources",
        "Police reports (how to file/obtain), incident records",

        "Traffic, municipal, small claims filing",
        "Protection/restraining orders (process, forms)",
        "Jury duty",
        "Fines/fees payment plans",
        "Expungement & record sealing (state-specific)",
        "Notary commissions and apostilles",

        "Birth/death/marriage certificates",
        "Marriage licenses and civil unions",
        "Name change process",
        "Domestic partnership registration (where applicable)",
        "Public records/FOIA (state public records laws)",
        "Proof of residency letters (when available)",

        "Rental licensing and inspections",
        "Habitability/code complaints",
        "Fair housing & anti-discrimination enforcement",
        "Security deposit rules (state)",
        "Eviction diversion/mediation, rental assistance",
        "Short-term rental compliance & taxes",

        "State income tax filing, refunds (where applicable)",
        "Property tax billing, exemptions (homestead, veterans, seniors)",
        "Sales/use/lodging taxes; marketplace rules",
        "Local business taxes & renewals",
        "Payment plans and penalties",
        "Unclaimed property searches & claims",

        "Voter registration, deadlines, ID requirements",
        "Polling place/ballot drop box lookup",
        "Absentee/mail-in ballots",
        "Sample ballots, measures, nonpartisan voter guides",
        "Poll worker signup",

        "Facility rentals (parks, shelters, fields)",
        "Special event/film permits",
        "Youth/senior programs and scholarships",
        "Hunting/fishing licenses, state park passes",
        "Library cards, fines, room bookings",

        "Pet licensing, rabies/vaccine clinics",
        "Animal control, bites, dangerous dogs",
        "Livestock/urban agriculture permits (chickens, bees)",
        "Wildlife conflicts and rehab",
        "4-H/extension programs; farm permits & scales",

        "Complaints to Attorney General or consumer office",
        "Price gouging, scams, robocalls",
        "Wage theft & minimum wage ordinances (local/state)",
        "Human rights commissions (housing/employment/public accommodations)",

        "Unemployment insurance claims and appeals",
        "Paid family/medical leave (state programs)",
        "Workplace safety (state OSHA where applicable)",
        "Apprenticeship registration",

        "Small business grants/loans; façade programs",
        "Opportunity zones, tax abatements",
        "Land bank properties",
        "Main Street & BID contacts",

        "ADA accommodations for facilities and programs",
        "Language access & interpretation/translation",
        "Accessible web/app services and alternative formats",
        "Assistive technology loan programs",

        "Which agency handles X?” routing",
        "Eligibility screening (“Do I qualify?”)",
        "Document checklists",
        "Timelines/SLAs and fee calculators",
        "How to appeal/renew/reinstate",
        "Contact info, office hours, appointments"
    ]

    # Two-stage prompting
    QUESTION_PROMPT: str = (
        "Generate {NUM_QUESTIONS} QUESTIONs. The questions must be on this topic - {TOPIC}.\n\n"
        "The questions should be phrased in a way it's coming from US citizens about real problems they are having in varied tone and emotion depending on the question"
        "The question is being posed to a government agency to get help solving their problem on "
        "the topic {TOPIC}."
    )

    ANSWER_PROMPT: str = (
        "You are CivAssist, a warm, trustworthy government guide that helps residents navigate "
        "public services. Your mission is to reduce friction and increase successful outcomes by "
        "giving accurate, current, plain-language guidance, plus concrete next steps.\n\n"
        "Core Behaviors:\n"
        "1. Be welcoming & calm. Start with a brief summary of the user's goal in your own words.\n"
        "2. Get them unstuck. Provide actionable next steps (forms, eligibility checks, fees, deadlines, where to go, who to call).\n"
        "3. Plain language. Short sentences. Avoid jargon; if you must use a legal term, define it.\n"
        "4. Structured outputs. Use headings, bullets, and numbered steps. Put critical info (deadlines, fees, documents) in tidy lists.\n"
        "5. Safety & accuracy first. If information is time-sensitive, jurisdiction-specific, or likely to change, explicitly verify it (and say when something may vary by region).\n"
        "6. Respect intent. If the user is venting or worried, acknowledge feelings briefly, then move to solutions.\n"
        "7. Minimize burden. Where possible, do eligibility triage, calculate fees/penalties, pre-fill checklists, and provide contact scripts.\n"
        "8. Privacy by default. Never ask for sensitive identifiers (full SSN/Aadhaar, full DOB) unless strictly necessary; if needed, explain why and how it's used.\n"
        "9. Accessibility. Offer alternatives (in-person, phone, mail). Provide step-by-step directions and required documents in checklists.\n"
        "10. Fairness. Avoid bias; present options neutrally and disclose trade-offs.\n"
        "11. Avoid Fabrication\n\n"
        
        "PROCEDURE:\n"
        "1.Think comprehensively what {NUM_OF_VIEWPOINTS} VIEWPOINTS are pertinent to the question - {QUESTION}.\n\n"
        "2.For each VIEWPOINT, explain it comprehensively and use relatable examples to support your answer.\n\n"
        "3. Finally combine these {NUM_OF_VIEWPOINTS} VIEWPOINTS above into a coherent prose ANSWER. "
        "4. Fact check your answer in detail. REDO if it there are factual untruths or mistakes.\n\n"
        "5. The number of words per ANSWER should be approximately {TARGET_LEN_OF_ANS}.\n\n"
    )

    NUM_QUESTIONS: int = 5  # number of questions per topic

    PARALLELISM: int = 2  # e.g., 2 or 5
    FLUSH_EVERY_N: int = 10  # e.g., 50 for large runs (Q/A pairs)

    OUTPUT_DIR: str = os.path.join(os.path.dirname(__file__), "output")
    OUTPUT_FILENAME: str = "synthetic_data.jsonl"  # JSONL with one object per line
    PROGRESS_FILENAME: str = "covered_topics.txt"  # one topic per line

    # Azure OpenAI settings
    AZURE_DEPLOYMENT: str = "gpt-4o-mini"  # set to your Azure deployment name
    TEMPERATURE: float = 0.3

    run(
        topics=TOPICS,
        question_prompt=QUESTION_PROMPT,
        answer_prompt=ANSWER_PROMPT,
        num_questions=NUM_QUESTIONS,
        parallelism=PARALLELISM,
        flush_every_n=FLUSH_EVERY_N,
        output_dir=OUTPUT_DIR,
        output_filename=OUTPUT_FILENAME,
        progress_filename=PROGRESS_FILENAME,
        azure_deployment=AZURE_DEPLOYMENT,
        temperature=TEMPERATURE,
    )
