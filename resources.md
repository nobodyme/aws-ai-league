Materials used to learn and participate in fine-tuning

## Resources

### About LLM league
- https://aws.amazon.com/blogs/aws/aws-ai-league-learn-innovate-and-compete-in-our-new-ultimate-ai-showdown/
- https://aws.amazon.com/blogs/publicsector/aws-llm-league-sparks-ai-innovation-at-aws-summit-washington-dc-2025/

### Jumpstart examples
- https://github.com/aws/amazon-sagemaker-examples/tree/main/introduction_to_amazon_algorithms/jumpstart-foundation-models

### Youtube
- Video tutorial series amazon - https://www.youtube.com/watch?v=gZP9N86b248&list=PLBm5soQMjeJ0-h5Dfp_iUFyjUDDbxM8S4&index=6 - Done
    - Quantization improves speed - What types of quantizations are there? (On contrary I only read it reduces memory and so slows down training time) TODO
    - Use `ml.g5.48xlarge`
- https://www.youtube.com/watch?v=d4Cnv_g3GiI
- https://www.youtube.com/watch?v=j6dqO2dSF9c

### Previous LLM success stories
- https://medium.com/@andyphuawc/my-secret-sauce-for-the-inaugural-singapore-nationwide-aws-large-language-models-league-llml-983d02e63cb3
    - Model used - `Llama-3-8B-Instruct` or 3.2 with LLM judge being a `Llama-3–70B-Instruct model` on 49 undisclosed questions
    - Final round judged by a LLM (40%), a panel of five experts (40%), and audience (20%)
    - Trial and error with params,
        epoch: 1 to 5 (with one training iteration using 10)
        learning_rate: 0.00005, 0.00002, 0.0001, 0.0002, 0.0003 (with one training iteration using 0.001)
        lora_r: 4, 8, 16, 256
        lora_alpha: 8, 32, 128, 256, 512
    - With lora_alpha being 2x of lora_r
    - Kept epoch at 1 - seeing instruction fine tuned model overfit after more epochs
        ```
        Final parameter set
            - epoch: 2
            - learning_rate: 0.0001
            - lora_r: 4
            - lora_alpha: 8
        ```
    - Only trained with 1000 odd examples initially
    - Looks like one will be given `only 3 training hours` - (although the training time could potentially be reduced using a higher per_device_train_batch_size as the default is 1, but there is a need to tread this carefully to avoid out-of-memory issue)
    - Used party rock app and refined output using, but ultimately used [dataset from github](https://github.com/SeaEval/SeaEval) for submission
        ```
        <Paste the instruction-context-reponse generated from Partyrock here>
        In the above, "instruction" refers to a question and "response" refers to the answer to the question. 
        Evaluate how accurate and comprehensive the "response" is to the "instruction". 
        This will be used to fine tune a large language model on Singapore culture. 
        Replace the "response" with one that is more accurate and comprehensive if required. 
        Output in the same format, each response on a new line but without any new blank line between responses, in json format.
        ```
    - https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms
        - Asks to enable lora for other params as well, query, key, value, projection, mlp, head
        - r too large or epoch too large could result in overfitting but suggests that diverse dataset training might require a bigger r value
        - Achieved best result with r = 256 and alpha = 128 (0.5 fold scaling as opposed to conventional 2x scaling)
    - https://magazine.sebastianraschka.com/p/ahead-of-ai-9-llm-tuning-and-dataset
    - [Less is more](https://arxiv.org/pdf/2305.11206)
    - [Long is more](https://arxiv.org/pdf/2402.04833)
    - [Random is all you need](https://arxiv.org/pdf/2410.09335)
    - The extremely simple baseline of selecting the 1,000 instructions with longest responses — that intuitively contain more learnable information and are harder to overfit — from standard datasets can consistently outperform these sophisticated methods
    - In the end however, ended up using `SeaEval-26k` instruction set, yeilding the best score
    `- Question: Use the training and evaluation losses as indication of overfitting - How to do this?`
    - Prompt used in final round
        ```
        You are an expert on Singapore culture and will be judged on your answer. 
        Answer the question as accurate and comprehensive. 
        Start with a title and then a short introduction. 
        Use sub paragraphs and elaborate in details in point forms. 
        Give as many details and examples as possible.
        ```
    - Prompt used for final question
        ```
        You are an expert on Singapore culture and will be judged on your answer. 
        Answer the question as accurate and comprehensive.
        Come up with an acronym and explain in details.
        Be creative and funny.
        ```

- https://github.com/taswhe/2024-sg-lol-papaoutai - preliminary round winner
    - Used https://partyrock.aws/u/papaoutai/ovCF-g4rS/QnACrafter/ (custom made fact checked by 4o mini) and datasets prepared by 4o
    ![parameters](image.png)
    - TODO: What's lora_dropout, lora_modules?
    - Used datasize of 1708 of synthetic data, came first
- https://medium.com/@jiaweilin02/how-to-excel-in-the-large-language-models-league-llml-at-aws-re-invent-2024-8f6d8606ebd6
    - Prelims:
        1) Use Python code to clean datasets, generate training data (qn & ans pair)
        2) Use ChatGPT to analyze hyperparamaters and datasets
        3) Create agentic workflow comprising of fact checker, rewriter and expert to further optimize datasets
        4) Use chain-of-thought prompting technique to generate datasets
        5) Use markdown format when crafting prompts to generate datasets
        6) Use AWS PartyRock to generate huge number of topics (eg. 20) and for each topic, generate large number of qns & ans pair (eg. 50)
        7) Experiment with different permutations of epoch and learning rate, tabulate the results in an excel sheet to find out what is the best combination (do take note that varying data record lines does affect epoch and learning rate so there is no one-size fits all)
        8) Study research papers available online that are related to LLM fine-tuning
    - Final:
        1) Craft a prompt template before the finale and insert the question into the prompt template to ask the Gen AI model (eg. ChatGPT) to generate the prompt
        2) Structure prompt to generate responses to be as long as possible (by adjusting number of tokens to be maximum) to score well for LLM judge, but potentially suffer at human panel of judges (as the response was simply too long and looked gibberish)
        3) Use markdown format in prompts
        4) Vary the temperature accordingly to the question, if the answer needed to be as definite as possible, lower the temperature eg. 0.2 to 0.5. If the answer is open-ended and needed to be creative, increase the temperature eg. 0.8 to 1.
        5) Top-P — Controls how many different words or phrases the language model considers when it’s trying to predict the next word. Top-P computes the cumulative probability distribution, and cut off as soon as that distribution exceeds the value of Top-P. For example, a Top-P of 0.3 means that only the tokens comprising the top 30% probability mass are considered.
- https://towardsaws.com/inside-the-aws-llm-league-lessons-from-fine-tuning-llms-for-a-citys-311-service-da054b3ee06b (Competition within Amazon)
    - Used the QnACrafter from before but mentions that context field like, hence model did not respond like a 311 representative but gave generalized answers
    ```You are a 311 call center agent. You received a question from a citizen of your city, and you should respond in a friendly manner and be as straightforward as possible.```
    - lora_r (rank) ranges from 4 to 256, with 8, 16, and 32 being common choices (determines the number of trainable parameters in the adaptation layers - high value meaning longer training time and more adaptability)
    - lora_alpha is a scaling factor that controls the magnitude of the LoRA weight updates, controls impact of adaptations (2x lora_r)
    - An epoch is a hyperparameter representing one complete pass through the entire training dataset during the model training process, recommends just 1 epoch
    - Learning rate - recommends 0.00002

### Other Synthetic Data Generation tools
- https://partyrock.aws/u/TheRayG/PmL1RViBp/Simple-AWS-LLMs-League-Dataset-Generator
- https://partyrock.aws/u/TheRayG/IInyME_vt/Advanced-AWS-LLMs-League-Dataset-Generator
- https://partyrock.aws/u/JiaweiLin/IvPiedcHN/LLMs-Datasets-Generator
- https://github.com/meta-llama/synthetic-data-kit
- https://github.com/meta-llama/synthetic-data-kit/tree/main/use-cases/awesome-synthetic-data-papers

### Reddit posts
- https://www.reddit.com/r/LocalLLaMA/comments/17p6hup/beginners_guide_to_finetuning_llama_2_and_mistral/
- https://huggingface.co/blog/4bit-transformers-bitsandbytes
- https://rentry.org/llm-training

### OpenAI
- https://platform.openai.com/docs/guides/model-optimization?optimization_videos=cost

### Amazon Skill Builder
- https://skillbuilder.aws/learn/CDYTAJCKGY/optimizing-foundation-models/PVR1FRGN1T
- https://skillbuilder.aws/learn/9V57WCXR4B/aws-simulearn-finetuning-an-llm-on-amazon-sagemaker/W3HW8X5NDX
- https://skillbuilder.aws/learn/AJY3PDJ476/aws-simulearn-automate-finetuning-of-an-llm/EEZRBZXXW9
- https://skillbuilder.aws/learn/5FK1N9FES7/advanced-finetuning-methods-on-amazon-sagemaker-ai/XE2SGGVR2Y

### Amazon Blogs
- https://aws.amazon.com/blogs/machine-learning/fine-tune-llama-3-for-text-generation-on-amazon-sagemaker-jumpstart/ (Exact LLM league usecase and shows exact timetaken for each configuration and exact hyperparameters)
    - Suggested perfect combination for 8B LLama variant
    - Instance Type        : ml.g5.48xlarge
    - Max Input Length     : 2048
    - Per Device Batch Size: 2 
    - Int8 Quantization    : False 
    - Enable FSDP          : True 
    - Time Taken (Minutes) : 27
- [Prompting technique for llama](https://aws.amazon.com/blogs/machine-learning/best-prompting-practices-for-using-meta-llama-3-with-amazon-sagemaker-jumpstart/)
- https://aws.amazon.com/blogs/machine-learning/customize-small-language-models-on-aws-with-automotive-terminology/ (Similar but without using jumpstart, good eval piece)
    - [Code example](https://github.com/aws-samples/customize-llm-automotive-aws)
- https://aws.amazon.com/blogs/machine-learning/operationalize-llm-evaluation-at-scale-using-amazon-sagemaker-clarify-and-mlops-services/
- https://aws.amazon.com/blogs/machine-learning/llm-as-a-judge-on-amazon-bedrock-model-evaluation/
- https://aws.amazon.com/sagemaker/ai/clarify/ (Should check)
- https://aws.amazon.com/blogs/machine-learning/advanced-fine-tuning-methods-on-amazon-sagemaker-ai/
- https://aws.amazon.com/blogs/machine-learning/fine-tune-large-language-models-with-reinforcement-learning-from-human-or-ai-feedback/
- https://aws.amazon.com/blogs/machine-learning/improving-your-llms-with-rlhf-on-amazon-sagemaker/
- https://aws.amazon.com/blogs/machine-learning/align-meta-llama-3-to-human-preferences-with-dpo-amazon-sagemaker-studio-and-amazon-sagemaker-ground-truth/
- https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html

### SLED datasets (Probably not yet accurate)
- https://docs.google.com/spreadsheets/d/1J5cVtyHfDps1RV0nYhHrzv40UD1mBEFb-oDZ2tg7Res/edit?usp=sharing
- https://media.usafacts.org/m/4ef126ccf2824bec/original/USAFacts_2025-FINAL.pdf

### Other potential
- https://iq.govwin.com/neo/marketAnalysis/index?=researchMarket=PSMAP
- https://sleds.mn.gov/
- https://labelstud.io/blog/five-open-dataset-resources-for-ai-training/
- https://cloud.google.com/datasets
- https://registry.opendata.aws/
- https://www.kaggle.com/datasets?tags=11105-Education
- https://catalog.data.gov/dataset/?q=&sort=views_recent+desc&groups=local&organization_type=State+Government&tags=education&ext_location=&ext_bbox=&ext_prev_extent=&page=2
